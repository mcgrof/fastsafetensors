# SPDX-License-Identifier: Apache-2.0
"""DMA-BUF copier: NVMe reads straight into device memory over io_uring.

The copier allocates the device memory for a shard, exports it as DMA-BUFs,
registers each DMA-BUF with io_uring as a fixed buffer bound to the
checkpoint's file descriptor, and issues ``READ_FIXED`` for the 4 KiB-aligned
superset of the tensor data. The NVMe controller writes into GPU memory
directly: no host memory carries the payload and no device-side copy follows,
because tensors are views into the loaded memory exactly as with ``gds``.

The kernel registers at most ``DMABUF_REGISTER_MAX_BYTES`` (1 GiB) per
DMA-BUF, and the ROCm exporter exports the whole driver allocation that
contains a pointer. A shard is therefore loaded into *pieces*: each piece is
its own runtime allocation of at most the cap, holding whole tensors. A
tensor larger than the cap gets a piece of its own size; it is read through
staging allocations of at most the cap and then copied device to device.
That copy is the only extra traffic the cap costs; a kernel without the cap
(``FASTSAFETENSORS_DMABUF_REGISTER_BYTES`` raised accordingly) loads every
shard as one piece.

Requirements:

- a kernel with DMA-BUF io_uring fixed buffers (the ``dmabuf-size-ceiling``
  series), the checkpoint on ext4, XFS or a raw block device on PCIe NVMe,
  and ``O_DIRECT`` accepted for the file;
- a DMA-BUF exporter for the device allocation: ``hsa_amd_portable_export_dmabuf``
  on ROCm or ``cuMemGetHandleForAddressRange`` on the NVIDIA open kernel
  modules.

On a CPU device the same code registers ordinary host memory, which keeps the
submission path testable without a GPU.

Environment variables:

- ``FASTSAFETENSORS_DMABUF_IO_BYTES``: bytes per ``READ_FIXED`` (default 16 MiB).
- ``FASTSAFETENSORS_DMABUF_QUEUE_DEPTH``: reads kept in flight (default 32).
- ``FASTSAFETENSORS_DMABUF_REGISTER_BYTES``: per-registration cap (default 1 GiB).
- ``FASTSAFETENSORS_DMABUF_STRICT=1``: fail instead of falling back to the
  ``nogds`` copier when the ring, the export or the file setup is refused.
"""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from .. import cpp as fstcpp
from ..allocation import SharedDeviceAllocation
from ..common import SafeTensorsMetadata, init_logger, is_gpu_found, is_odirect_enabled
from ..frameworks import FrameworkOpBase, TensorBase
from ..io_uring_dmabuf import (
    DMABUF_REGISTER_MAX_BYTES,
    IoUring,
    RegisteredRegion,
    dedicated_malloc,
    device_memcpy,
    register_destination,
    release_regions,
)
from ..st_types import Device, DeviceType, DType
from .base import (
    CopierInterface,
    validated_byte_ranges,
    validated_chunk_allocation_size,
)
from .nogds import load_library_func, new_nogds_file_copier
from .registry import CopierConstructFunc, register_copier_constructor

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)

ALIGN = 4096
_DEFAULT_IO_BYTES = 16 << 20
_DEFAULT_QUEUE_DEPTH = 32
_warned_fallback = False


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def strict_mode() -> bool:
    return os.environ.get("FASTSAFETENSORS_DMABUF_STRICT", "0") == "1"


def _round_down(x: int) -> int:
    return x - x % ALIGN


def _round_up(x: int) -> int:
    return (x + ALIGN - 1) // ALIGN * ALIGN


class _SetupRefused(Exception):
    """Export, registration or file setup was refused; nogds can still run."""


class _Piece:
    """One runtime allocation holding whole tensors of a shard."""

    __slots__ = (
        "file_lo",
        "file_hi",
        "names",
        "oversize",
        "gbuf",
        "pad",
        "regions",
        "staging",
    )

    def __init__(self, file_lo: int, file_hi: int):
        self.file_lo = file_lo  # aligned file offset of the first byte read
        self.file_hi = file_hi  # aligned end
        self.names: Set[str] = set()
        self.oversize = False
        self.gbuf: Optional[fstcpp.gds_device_buffer] = None
        self.pad = 0  # bytes between the allocation base and the I/O start
        self.regions: List[RegisteredRegion] = []
        # (gbuf, pad, file_off, length, regions) staging buffers of an
        # oversize piece; copied into gbuf after completion
        self.staging: List[Tuple[fstcpp.gds_device_buffer, int, int, int, List]] = []

    @property
    def span(self) -> int:
        return self.file_hi - self.file_lo

    def dest(self) -> int:
        assert self.gbuf is not None
        return self.gbuf.get_base_address() + self.pad


class DmabufFileCopier(CopierInterface):
    def __init__(
        self,
        metadata: SafeTensorsMetadata,
        device: Device,
        ring: IoUring,
        framework: FrameworkOpBase,
        io_bytes: int = _DEFAULT_IO_BYTES,
        queue_depth: int = _DEFAULT_QUEUE_DEPTH,
        register_bytes: int = DMABUF_REGISTER_MAX_BYTES,
        fallback_cache: Optional[List[CopierConstructFunc]] = None,
    ):
        self.framework = framework
        self.metadata = metadata
        self.device = device
        self.ring = ring
        self.io_bytes = _round_down(max(io_bytes, ALIGN))
        self.queue_depth = max(1, queue_depth)
        self.register_bytes = _round_down(max(register_bytes, 2 * ALIGN))
        self.is_device = device.type in (DeviceType.CUDA, DeviceType.GPU)
        self.is_hip = bool(fstcpp.is_hip_found()) if self.is_device else False
        self.fd = -1
        self.pieces: List[_Piece] = []
        self.reqs: List[Tuple[int, int, int]] = []  # (user_data, file_off, length)
        self.byte_ranges: Optional[List[Tuple[int, int]]] = None
        self._chunk_names: Optional[Set[str]] = None
        self._chunk_allocation_size: Optional[int] = None
        self._fallback: Optional[CopierInterface] = None
        self._fallback_cache = fallback_cache
        self.staging_copy_bytes = 0

    # -- range selection ----------------------------------------------------

    def set_byte_ranges(self, byte_ranges: Optional[List[Tuple[int, int]]]) -> None:
        self.byte_ranges = validated_byte_ranges(self.metadata, byte_ranges)

    def set_chunk(
        self,
        byte_ranges: List[Tuple[int, int]],
        names: Set[str],
        allocation_size: Optional[int] = None,
    ) -> None:
        checked = validated_byte_ranges(self.metadata, byte_ranges)
        assert checked is not None
        self.byte_ranges = checked
        self._chunk_names = names
        self._chunk_allocation_size = validated_chunk_allocation_size(
            checked, allocation_size
        )

    @classmethod
    def chunk_transient_multiplier(cls, paths: List[str]) -> int:
        # Oversize tensors are staged once more; whole-tensor pieces are not.
        return 2

    @classmethod
    def chunk_device_overhead(cls, paths: List[str]) -> int:
        return 3 * ALIGN

    @classmethod
    def fixed_device_overhead(cls, paths: List[str]) -> int:
        return 0

    # -- fallback -----------------------------------------------------------

    def _use_fallback(self, reason: str) -> CopierInterface:
        global _warned_fallback
        if strict_mode():
            raise RuntimeError(f"dmabuf copier: {reason}")
        if not _warned_fallback:
            _warned_fallback = True
            logger.warning(
                "dmabuf copier unavailable (%s); falling back to nogds", reason
            )
        if self._fallback_cache is not None:
            if not self._fallback_cache:
                self._fallback_cache.append(
                    new_nogds_file_copier(self.device, framework=self.framework)
                )
            fallback = self._fallback_cache[0](
                self.metadata, self.device, self.framework
            )
        else:
            fallback = new_nogds_file_copier(self.device, framework=self.framework)(
                self.metadata, self.device, self.framework
            )
        if self._chunk_names is not None:
            assert self.byte_ranges is not None
            fallback.set_chunk(
                self.byte_ranges, self._chunk_names, self._chunk_allocation_size
            )
        else:
            fallback.set_byte_ranges(self.byte_ranges)
        self._fallback = fallback
        return fallback

    # -- planning -----------------------------------------------------------

    def _selected_tensors(self) -> List[Tuple[int, int, str]]:
        """``(start, end, name)`` in file order for the tensors to load."""
        header_length = self.metadata.header_length
        out = []
        for name, frame in self.metadata.tensors.items():
            start = header_length + frame.data_offsets[0]
            end = header_length + frame.data_offsets[1]
            if self._chunk_names is not None:
                if name not in self._chunk_names:
                    continue
            elif self.byte_ranges is not None and start < end:
                covered = start
                for lo, hi in self.byte_ranges:
                    if lo <= covered < hi:
                        covered = min(end, hi)
                        if covered == end:
                            break
                if covered != end:
                    continue
            out.append((start, end, name))
        out.sort()
        return out

    def _plan_pieces(self) -> List[_Piece]:
        cap_span = self.register_bytes - ALIGN  # allocation = span + pad page
        pieces: List[_Piece] = []
        zero_len: List[Tuple[int, str]] = []
        cur: Optional[_Piece] = None
        for start, end, name in self._selected_tensors():
            if start == end:
                zero_len.append((start, name))
                continue
            lo, hi = _round_down(start), _round_up(end)
            if cur is not None and hi - cur.file_lo <= cap_span:
                cur.file_hi = max(cur.file_hi, hi)
                cur.names.add(name)
                continue
            cur = _Piece(lo, hi)
            cur.names.add(name)
            cur.oversize = hi - lo > cap_span
            pieces.append(cur)
        for start, name in zero_len:
            for piece in pieces:
                if piece.file_lo <= start <= piece.file_hi:
                    piece.names.add(name)
                    break
            else:
                if pieces:
                    # A zero-length view needs no bytes; place it at the
                    # first piece's start so its offset stays in bounds.
                    pieces[0].names.add(name)
        if (
            self._chunk_allocation_size is not None
            and len(pieces) == 1
            and not pieces[0].oversize
        ):
            want = _round_up(self._chunk_allocation_size)
            if want <= cap_span:
                pieces[0].file_hi = max(pieces[0].file_hi, pieces[0].file_lo + want)
        return pieces

    def _runs_in(self, piece: _Piece) -> List[Tuple[int, int]]:
        """Aligned file ranges to read for ``piece``."""
        if self.byte_ranges is None:
            runs = [
                (piece.file_lo, min(piece.file_hi, _round_up(self.metadata.size_bytes)))
            ]
        else:
            runs = []
            for lo, hi in self.byte_ranges:
                lo, hi = max(_round_down(lo), piece.file_lo), min(
                    _round_up(hi), piece.file_hi
                )
                if hi <= lo:
                    continue
                if runs and lo <= runs[-1][1]:
                    runs[-1] = (runs[-1][0], max(runs[-1][1], hi))
                else:
                    runs.append((lo, hi))
        return runs

    # -- allocation ---------------------------------------------------------

    def _alloc(self, length: int) -> Tuple[fstcpp.gds_device_buffer, int]:
        """Allocate ``length`` plus one page; return the buffer and the pad."""
        total = length + ALIGN
        if self.is_device:
            ptr = dedicated_malloc(total, self.is_hip)
            gbuf = fstcpp.gds_device_buffer(ptr, total, True)
            self.framework.mem_used += total
        else:
            gbuf = self.framework.alloc_tensor_memory(total, self.device)
        pad = (-gbuf.get_base_address()) % ALIGN
        return gbuf, pad

    def _free(self, gbuf: fstcpp.gds_device_buffer) -> None:
        self.framework.free_tensor_memory(gbuf, self.device)

    # -- submission ---------------------------------------------------------

    def _queue_reads(
        self, runs: List[Tuple[int, int]], dest_base: int, file_base: int, regions
    ) -> None:
        """Queue reads of ``runs`` so file offset ``file_base`` lands at ``dest_base``."""
        region_idx = 0
        for start, end in runs:
            count = start
            while count < end:
                dest = dest_base + (count - file_base)
                while regions[region_idx].dev_end <= dest:
                    region_idx += 1
                region = regions[region_idx]
                req_len = min(end - count, self.io_bytes, region.dev_end - dest)
                while self.ring.inflight + self.ring.pending >= self.queue_depth:
                    self.ring.submit(wait_nr=1)
                user_data = self.ring.read_fixed(
                    self.fd,
                    count,
                    region.slot,
                    region.buf_off + (dest - region.dev_start),
                    req_len,
                )
                self.reqs.append((user_data, count, req_len))
                count += req_len

    def _register(self, dest: int, length: int) -> List[RegisteredRegion]:
        try:
            return register_destination(
                self.ring,
                dest,
                length,
                self.fd,
                self.is_device,
                self.is_hip,
                max_bytes=self.register_bytes,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise _SetupRefused(str(exc)) from exc

    def _submit_piece(self, piece: _Piece) -> None:
        runs = self._runs_in(piece)
        if not piece.oversize:
            piece.regions = self._register(piece.dest(), piece.span)
            self._queue_reads(runs, piece.dest(), piece.file_lo, piece.regions)
            return
        # Oversize: stage through registered allocations of at most the cap.
        stage_span = self.register_bytes - ALIGN
        for lo, hi in runs:
            off = lo
            while off < hi:
                n = min(stage_span, hi - off)
                sbuf, spad = self._alloc(n)
                try:
                    regions = self._register(sbuf.get_base_address() + spad, n)
                except BaseException:
                    self._free(sbuf)
                    raise
                piece.staging.append((sbuf, spad, off, n, regions))
                self._queue_reads(
                    [(off, off + n)], sbuf.get_base_address() + spad, off, regions
                )
                off += n

    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> fstcpp.gds_device_buffer:
        if max_copy_block_size <= 0:
            raise ValueError("max_copy_block_size must be positive")
        if self.is_device and not is_gpu_found():
            return self._use_fallback("GPU runtime library not found").submit_io(
                use_buf_register, max_copy_block_size
            )
        direct = is_odirect_enabled(self.metadata.src)
        if self.is_device and not direct:
            return self._use_fallback(
                f"{self.metadata.src} is not eligible for O_DIRECT"
            ).submit_io(use_buf_register, max_copy_block_size)
        flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
        try:
            self.fd = os.open(self.metadata.src, flags)
        except OSError as exc:
            if direct and exc.errno == errno.EINVAL:
                return self._use_fallback(
                    f"{self.metadata.src}: filesystem rejected O_DIRECT"
                ).submit_io(use_buf_register, max_copy_block_size)
            raise
        self.io_bytes = min(self.io_bytes, _round_down(max(max_copy_block_size, ALIGN)))
        pieces = self._plan_pieces()
        if not pieces:
            # Nothing selected: hand back one page so the factory has a buffer.
            gbuf, pad = self._alloc(0)
            empty = _Piece(_round_down(self.metadata.header_length), 0)
            empty.file_hi = empty.file_lo
            empty.gbuf, empty.pad = gbuf, pad
            self.pieces = [empty]
            os.close(self.fd)
            self.fd = -1
            return gbuf
        try:
            for piece in pieces:
                piece.gbuf, piece.pad = self._alloc(piece.span)
                self.pieces.append(piece)
                self._submit_piece(piece)
            self.ring.submit()
        except _SetupRefused as exc:
            self._abort()
            return self._use_fallback(str(exc)).submit_io(
                use_buf_register, max_copy_block_size
            )
        except BaseException:
            self._abort()
            raise
        return self.pieces[0].gbuf

    def _abort(self) -> None:
        """Drain, release and free everything; used before a retry or re-raise."""
        try:
            for user_data, _off, _length in self.reqs:
                try:
                    self.ring.wait(user_data)
                except Exception:
                    pass
        finally:
            self.reqs.clear()
            self._release_registrations()
            for piece in self.pieces:
                for sbuf, *_ in piece.staging:
                    self._free(sbuf)
                piece.staging.clear()
                if piece.gbuf is not None:
                    self._free(piece.gbuf)
                    piece.gbuf = None
            self.pieces.clear()

    def _release_registrations(self) -> Optional[Exception]:
        error: Optional[Exception] = None
        for piece in self.pieces:
            err = release_regions(self.ring, piece.regions)
            error = error or err
            for _sbuf, _spad, _off, _n, regions in piece.staging:
                err = release_regions(self.ring, regions)
                error = error or err
        if self.fd >= 0:
            try:
                os.close(self.fd)
            except OSError as exc:
                error = error or exc
            self.fd = -1
        return error

    # -- completion ---------------------------------------------------------

    def finish_io(self) -> None:
        if self._fallback is not None:
            try:
                self._fallback.finish_io()
            finally:
                self._fallback = None
            return
        failures: List[str] = []
        data_end = self.metadata.size_bytes
        try:
            for user_data, off, length in self.reqs:
                res = self.ring.wait(user_data)
                needed = max(0, min(data_end, off + length) - off)
                if res < 0:
                    failures.append(
                        f"offset={off} len={length}: {os.strerror(-res)} (errno {-res})"
                    )
                elif res < needed:
                    failures.append(
                        f"offset={off} len={length}: short read {res} < {needed}"
                    )
        finally:
            self.reqs.clear()
            error = self._release_registrations()
        if failures:
            hint = ""
            if any("errno 95" in f for f in failures):
                hint = (
                    " (EOPNOTSUPP: the file system or device has no DMA-BUF "
                    "io_uring support; only ext4, XFS and raw block devices on "
                    "PCIe NVMe qualify)"
                )
            self._free_staging()
            raise RuntimeError(
                f"dmabuf copier: {len(failures)} read(s) failed on "
                f"{self.metadata.src}{hint}: {failures[:4]}"
            )
        # Oversize pieces: gather the staged bytes into their final allocation.
        try:
            for piece in self.pieces:
                for sbuf, spad, off, n, _regions in piece.staging:
                    dst = piece.dest() + (off - piece.file_lo)
                    src = sbuf.get_base_address() + spad
                    if self.is_device:
                        device_memcpy(dst, src, n, self.is_hip)
                    else:
                        ctypes.memmove(dst, src, n)
                    self.staging_copy_bytes += n
        finally:
            self._free_staging()
        if error is not None:
            raise error

    def _free_staging(self) -> None:
        for piece in self.pieces:
            for sbuf, *_ in piece.staging:
                self._free(sbuf)
            piece.staging.clear()

    def wait_io(
        self,
        gbuf: fstcpp.gds_device_buffer,
        dtype: DType = DType.AUTO,
        noalign: bool = False,
        owner: Optional[SharedDeviceAllocation] = None,
    ) -> Dict[str, TensorBase]:
        if self._fallback is not None:
            tensors = self._fallback.wait_io(
                gbuf, dtype=dtype, noalign=noalign, owner=owner
            )
            self._fallback = None
            return tensors
        self.finish_io()
        pieces, self.pieces = self.pieces, []
        if not pieces:
            return {}
        assert pieces[0].gbuf is gbuf
        extras = pieces[1:]
        if owner is None and extras:
            for piece in extras:
                self._free(piece.gbuf)
            if noalign:
                # Factory cleanup after a failure elsewhere: nothing to build.
                return {}
            raise RuntimeError(
                "dmabuf copier: borrowed_tensors needs a single-piece load; "
                f"{self.metadata.src} needs {len(pieces)} pieces under the "
                f"{self.register_bytes} byte registration cap"
            )
        tensors: Dict[str, TensorBase] = {}
        for index, piece in enumerate(pieces):
            if index == 0:
                piece_owner = owner
            else:
                piece_owner = SharedDeviceAllocation(
                    piece.gbuf, self.framework, self.device
                )
            try:
                tensors.update(
                    self.metadata._get_tensors(
                        piece.gbuf,
                        self.device,
                        piece.file_lo - piece.pad,
                        dtype=dtype,
                        names=piece.names,
                        owner=piece_owner,
                    )
                )
            finally:
                if index > 0:
                    # Tensors hold their own references; drop the copier's.
                    piece_owner.release()
        return tensors


def _new_ring(queue_depth: int) -> IoUring:
    entries = 1
    while entries < queue_depth * 2:
        entries <<= 1
    return IoUring(entries=entries, buffer_slots=128)


@register_copier_constructor("dmabuf", DmabufFileCopier)
def new_dmabuf_file_copier(
    device: Device,
    framework: FrameworkOpBase,
    bbuf_size_kb: int = 16 * 1024,
    max_threads: int = 16,
    **kwargs,
) -> CopierConstructFunc:
    """Build DMA-BUF copiers sharing one io_uring instance per loader.

    Falls back to the ``nogds`` constructor when the platform cannot provide
    io_uring at all; per-file refusals (export, registration, O_DIRECT) fall
    back inside the copier so a loader keeps working on a mixed set of files.
    """

    def fallback(reason: str) -> CopierConstructFunc:
        if strict_mode():
            raise RuntimeError(f"dmabuf copier: {reason}")
        warnings.warn(
            f"dmabuf copier unavailable ({reason}); falling back to nogds",
            UserWarning,
        )
        return new_nogds_file_copier(
            device, bbuf_size_kb, max_threads, framework=framework, **kwargs
        )

    if platform.system() != "Linux":
        return fallback("io_uring needs Linux")
    load_library_func(framework)
    if device.type != DeviceType.CPU and not is_gpu_found():
        return fallback("GPU runtime library not found")
    io_bytes = _env_int("FASTSAFETENSORS_DMABUF_IO_BYTES", _DEFAULT_IO_BYTES)
    queue_depth = _env_int("FASTSAFETENSORS_DMABUF_QUEUE_DEPTH", _DEFAULT_QUEUE_DEPTH)
    register_bytes = _env_int(
        "FASTSAFETENSORS_DMABUF_REGISTER_BYTES", DMABUF_REGISTER_MAX_BYTES
    )
    try:
        ring = _new_ring(queue_depth)
    except OSError as exc:
        return fallback(f"io_uring setup failed: {exc}")

    fallback_cache: List[CopierConstructFunc] = []

    def construct_copier(
        metadata: SafeTensorsMetadata, device: Device, framework: FrameworkOpBase
    ) -> CopierInterface:
        return DmabufFileCopier(
            metadata,
            device,
            ring,
            framework,
            io_bytes=io_bytes,
            queue_depth=queue_depth,
            register_bytes=register_bytes,
            fallback_cache=fallback_cache,
        )

    # Keep the ring alive as long as the constructor is referenced by the loader.
    construct_copier.ring = ring  # type: ignore[attr-defined]
    return construct_copier
