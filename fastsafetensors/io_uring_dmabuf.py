# SPDX-License-Identifier: Apache-2.0
"""Minimal io_uring submission layer for DMA-BUF and host fixed buffers.

The dmabuf copier needs three things from the kernel: a ring, a fixed-buffer
table whose slots can hold a DMA-BUF bound to one target file, and
``IORING_OP_READ_FIXED`` submissions against those slots. liburing does not
expose the DMA-BUF registration descriptor yet, so this module talks to the
raw system calls through ``ctypes``. It deliberately covers nothing else.

The registration ABI is the io_uring DMA-BUF registered-buffer series queued
for Linux 7.4: ``io_uring_register(IORING_REGISTER_BUFFERS_UPDATE)`` with
``IORING_RSRC_UPDATE_EXTENDED`` and a descriptor of type
``IO_REGBUF_TYPE_DMABUF`` that carries the DMA-BUF fd and the file descriptor
whose device will DMA into it; ``uaddr`` and ``size`` must be zero. A
``READ_FIXED`` into such a slot addresses the buffer by byte offset from the
start of the DMA-BUF, not by a user pointer. That series supports raw NVMe
block devices; reads from ext4 or XFS files need file system support that is
not upstream yet.

Host memory is registered through the ordinary ``iovec`` update so the same
ring and the same copier run on a CPU-only host, where the tests live.

The ring is shared with the kernel through ``mmap``. Indices the kernel
writes are read with acquire loads and indices the kernel reads are written
with release stores, through helpers in the C++ extension, because plain
Python memory access does not order them on weakly ordered CPUs such as
arm64.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import functools
import mmap
import os
import struct
from typing import Dict, Iterator, List, Optional, Tuple

from . import cpp as fstcpp

# ---------------------------------------------------------------------------
# UAPI constants (include/uapi/linux/io_uring.h)
# ---------------------------------------------------------------------------
_SYS_IO_URING_SETUP = 425
_SYS_IO_URING_ENTER = 426
_SYS_IO_URING_REGISTER = 427

IORING_OP_READ_FIXED = 4

IORING_SETUP_CQSIZE = 1 << 3

IORING_ENTER_GETEVENTS = 1 << 0

IORING_OFF_SQ_RING = 0
IORING_OFF_CQ_RING = 0x8000000
IORING_OFF_SQES = 0x10000000

IORING_REGISTER_BUFFERS2 = 15
IORING_REGISTER_BUFFERS_UPDATE = 16
IORING_RSRC_REGISTER_SPARSE = 1 << 0
IORING_RSRC_UPDATE_EXTENDED = 1 << 1
IO_REGBUF_TYPE_DMABUF = 2

IORING_FEAT_SINGLE_MMAP = 1 << 0

# Every DMA-BUF io_uring kernel registers at least this many bytes per
# DMA-BUF; early versions of the series refuse anything larger.
DMABUF_REGISTER_MAX_BYTES = 1 << 30

# The series queued for Linux 7.4 refuses a DMA-BUF larger than UINT_MAX
# bytes, because the offset into it travels in a 32-bit bvec field. The ROCm
# exporter exports the whole driver allocation, which it rounds to 2 MiB, so
# the largest allocation that stays under the limit is 4 GiB - 2 MiB.
DMABUF_REGISTER_LARGE_BYTES = (1 << 32) - (2 << 20)

_SQE_SIZE = 64
_CQE_SIZE = 16
_PAGE = os.sysconf("SC_PAGE_SIZE")

_libc = ctypes.CDLL(None, use_errno=True)
_syscall = _libc.syscall
_syscall.restype = ctypes.c_long


def _errno_error(what: str) -> OSError:
    err = ctypes.get_errno()
    return OSError(err, f"{what}: {os.strerror(err)}")


class _IoUringParams(ctypes.Structure):
    _fields_ = [
        ("sq_entries", ctypes.c_uint32),
        ("cq_entries", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("sq_thread_cpu", ctypes.c_uint32),
        ("sq_thread_idle", ctypes.c_uint32),
        ("features", ctypes.c_uint32),
        ("wq_fd", ctypes.c_uint32),
        ("resv", ctypes.c_uint32 * 3),
        # struct io_sqring_offsets
        ("sq_head", ctypes.c_uint32),
        ("sq_tail", ctypes.c_uint32),
        ("sq_ring_mask", ctypes.c_uint32),
        ("sq_ring_entries", ctypes.c_uint32),
        ("sq_flags", ctypes.c_uint32),
        ("sq_dropped", ctypes.c_uint32),
        ("sq_array", ctypes.c_uint32),
        ("sq_resv1", ctypes.c_uint32),
        ("sq_user_addr", ctypes.c_uint64),
        # struct io_cqring_offsets
        ("cq_head", ctypes.c_uint32),
        ("cq_tail", ctypes.c_uint32),
        ("cq_ring_mask", ctypes.c_uint32),
        ("cq_ring_entries", ctypes.c_uint32),
        ("cq_overflow", ctypes.c_uint32),
        ("cq_cqes", ctypes.c_uint32),
        ("cq_flags", ctypes.c_uint32),
        ("cq_resv1", ctypes.c_uint32),
        ("cq_user_addr", ctypes.c_uint64),
    ]


class _RsrcRegister(ctypes.Structure):
    _fields_ = [
        ("nr", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("resv2", ctypes.c_uint64),
        ("data", ctypes.c_uint64),
        ("tags", ctypes.c_uint64),
    ]


class _RsrcUpdate2(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("data", ctypes.c_uint64),
        ("tags", ctypes.c_uint64),
        ("nr", ctypes.c_uint32),
        ("resv2", ctypes.c_uint32),
    ]


class _RegbufDesc(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("size", ctypes.c_uint64),
        ("uaddr", ctypes.c_uint64),
        ("dmabuf_fd", ctypes.c_int32),
        ("target_fd", ctypes.c_int32),
        ("resv", ctypes.c_uint64 * 6),
    ]


class _Iovec(ctypes.Structure):
    _fields_ = [("iov_base", ctypes.c_void_p), ("iov_len", ctypes.c_size_t)]


def _u32_at(buf: mmap.mmap, off: int) -> int:
    return struct.unpack_from("=I", buf, off)[0]


def _u32_set(buf: mmap.mmap, off: int, value: int) -> None:
    struct.pack_into("=I", buf, off, value & 0xFFFFFFFF)


class IoUring:
    """One io_uring instance with a sparse fixed-buffer table.

    The ring uses the legacy SQ index array so it runs on every kernel that
    has ``READ_FIXED``; nothing here needs ``IORING_SETUP_NO_SQARRAY``.
    Completions are reaped in whatever order the kernel posts them and are
    keyed by ``user_data``.
    """

    def __init__(self, entries: int = 64, buffer_slots: int = 64):
        if entries <= 0 or buffer_slots <= 0:
            raise ValueError("entries and buffer_slots must be positive")
        params = _IoUringParams()
        params.flags = IORING_SETUP_CQSIZE
        params.cq_entries = entries * 2
        fd = _syscall(_SYS_IO_URING_SETUP, ctypes.c_uint(entries), ctypes.byref(params))
        if fd < 0:
            raise _errno_error("io_uring_setup")
        self.fd = int(fd)
        self._closed = False
        try:
            # sq_entries/cq_entries are counts; the *_ring_entries fields are
            # offsets of the count inside the ring mapping, not counts.
            sq_size = params.sq_array + params.sq_entries * 4
            cq_size = params.cq_cqes + params.cq_entries * _CQE_SIZE
            if params.features & IORING_FEAT_SINGLE_MMAP:
                size = max(sq_size, cq_size)
                self._sq_mm = mmap.mmap(
                    self.fd,
                    size,
                    flags=mmap.MAP_SHARED | mmap.MAP_POPULATE,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                    offset=IORING_OFF_SQ_RING,
                )
                self._cq_mm = self._sq_mm
            else:
                self._sq_mm = mmap.mmap(
                    self.fd,
                    sq_size,
                    flags=mmap.MAP_SHARED | mmap.MAP_POPULATE,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                    offset=IORING_OFF_SQ_RING,
                )
                self._cq_mm = mmap.mmap(
                    self.fd,
                    cq_size,
                    flags=mmap.MAP_SHARED | mmap.MAP_POPULATE,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                    offset=IORING_OFF_CQ_RING,
                )
            self._sqes = mmap.mmap(
                self.fd,
                params.sq_entries * _SQE_SIZE,
                flags=mmap.MAP_SHARED | mmap.MAP_POPULATE,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                offset=IORING_OFF_SQES,
            )
        except BaseException:
            os.close(self.fd)
            raise
        # The extension's ordered accesses need the mappings' addresses; the
        # anchors keep the buffers exported until close() drops them.
        self._sq_anchor: Optional[ctypes.c_char] = ctypes.c_char.from_buffer(
            self._sq_mm
        )
        self._cq_anchor: Optional[ctypes.c_char] = ctypes.c_char.from_buffer(
            self._cq_mm
        )
        self._sq_base = ctypes.addressof(self._sq_anchor)
        self._cq_base = ctypes.addressof(self._cq_anchor)
        self._sq_head_off = params.sq_head
        self._sq_tail_off = params.sq_tail
        self._sq_mask = _u32_at(self._sq_mm, params.sq_ring_mask)
        self._sq_array_off = params.sq_array
        self._sq_entries = params.sq_entries
        self._cq_head_off = params.cq_head
        self._cq_tail_off = params.cq_tail
        self._cq_mask = _u32_at(self._cq_mm, params.cq_ring_mask)
        self._cq_cqes_off = params.cq_cqes
        self.pending = 0
        self.inflight = 0
        self._completions: Dict[int, int] = {}
        self._next_user_data = 1
        self.buffer_slots = buffer_slots
        self._free_slots: List[int] = list(range(buffer_slots))
        self._register_sparse(buffer_slots)

    # -- registration -------------------------------------------------------

    def _register(self, opcode: int, arg, nr_args: int) -> None:
        rc = _syscall(
            _SYS_IO_URING_REGISTER,
            ctypes.c_uint(self.fd),
            ctypes.c_uint(opcode),
            arg,
            ctypes.c_uint(nr_args),
        )
        if rc < 0:
            raise _errno_error(f"io_uring_register(opcode={opcode})")

    def _register_sparse(self, count: int) -> None:
        reg = _RsrcRegister(nr=count, flags=IORING_RSRC_REGISTER_SPARSE)
        self._register(IORING_REGISTER_BUFFERS2, ctypes.byref(reg), ctypes.sizeof(reg))

    def _update_slot(self, index: int, data, flags: int) -> None:
        up = _RsrcUpdate2(
            offset=index,
            flags=flags,
            data=ctypes.addressof(data),
            tags=0,
            nr=1,
            resv2=0,
        )
        self._register(
            IORING_REGISTER_BUFFERS_UPDATE, ctypes.byref(up), ctypes.sizeof(up)
        )

    def _take_slot(self) -> int:
        if not self._free_slots:
            raise RuntimeError("io_uring fixed-buffer table is full")
        return self._free_slots.pop(0)

    def register_dmabuf(self, dmabuf_fd: int, target_fd: int) -> int:
        """Bind a DMA-BUF to ``target_fd`` in a free slot and return the slot.

        The kernel sets up the DMA-BUF I/O context for ``target_fd`` here, so
        a file system or device without DMA-BUF support refuses the
        registration with ``EOPNOTSUPP``, and a DMA-BUF larger than the
        kernel accepts is refused with ``EINVAL``. The DMA mapping itself is
        built on the first read, which can still fail if the device cannot
        map the buffer.
        """
        index = self._take_slot()
        desc = _RegbufDesc(
            type=IO_REGBUF_TYPE_DMABUF,
            flags=0,
            size=0,
            uaddr=0,
            dmabuf_fd=dmabuf_fd,
            target_fd=target_fd,
        )
        try:
            self._update_slot(index, desc, IORING_RSRC_UPDATE_EXTENDED)
        except BaseException:
            self._free_slots.insert(0, index)
            raise
        return index

    def register_host(self, addr: int, length: int) -> int:
        """Register ``[addr, addr+length)`` of host memory in a free slot."""
        index = self._take_slot()
        iov = _Iovec(iov_base=addr, iov_len=length)
        try:
            self._update_slot(index, iov, 0)
        except BaseException:
            self._free_slots.insert(0, index)
            raise
        return index

    def unregister(self, index: int) -> None:
        """Clear a slot. The caller must have reaped every read that used it."""
        iov = _Iovec(iov_base=None, iov_len=0)
        self._update_slot(index, iov, 0)
        self._free_slots.append(index)

    # -- submission ---------------------------------------------------------

    def read_fixed(
        self, fd: int, file_off: int, slot: int, buf_addr: int, length: int
    ) -> int:
        """Queue ``READ_FIXED`` and return its ``user_data`` token.

        ``buf_addr`` is the byte offset into the DMA-BUF for a DMA-BUF slot,
        or the absolute host address for a host slot. ``submit()`` must follow.
        """
        if length <= 0 or length >= 1 << 32:
            raise ValueError(f"read length {length} out of range")
        tail = _u32_at(self._sq_mm, self._sq_tail_off)
        head = fstcpp.ring_load_acquire(self._sq_base + self._sq_head_off)
        if (tail - head) & 0xFFFFFFFF >= self._sq_entries:
            raise RuntimeError("submission queue is full; call submit() first")
        idx = tail & self._sq_mask
        user_data = self._next_user_data
        self._next_user_data += 1
        sqe = struct.pack(
            "=BBHiQQIIQHHiQQ",
            IORING_OP_READ_FIXED,  # opcode
            0,  # flags
            0,  # ioprio
            fd,
            file_off,  # off
            buf_addr,  # addr
            length,  # len
            0,  # rw_flags
            user_data,
            slot,  # buf_index
            0,  # personality
            0,  # splice_fd_in
            0,  # addr3
            0,  # __pad2
        )
        assert len(sqe) == _SQE_SIZE
        self._sqes[idx * _SQE_SIZE : (idx + 1) * _SQE_SIZE] = sqe
        _u32_set(self._sq_mm, self._sq_array_off + idx * 4, idx)
        fstcpp.ring_store_release(
            self._sq_base + self._sq_tail_off, (tail + 1) & 0xFFFFFFFF
        )
        self.pending += 1
        return user_data

    def submit(self, wait_nr: int = 0) -> int:
        """Hand queued entries to the kernel; optionally wait for completions."""
        to_submit = self.pending
        if to_submit == 0 and wait_nr == 0:
            return 0
        flags = IORING_ENTER_GETEVENTS if wait_nr > 0 else 0
        rc = _syscall(
            _SYS_IO_URING_ENTER,
            ctypes.c_uint(self.fd),
            ctypes.c_uint(to_submit),
            ctypes.c_uint(wait_nr),
            ctypes.c_uint(flags),
            None,
            ctypes.c_size_t(0),
        )
        if rc < 0:
            raise _errno_error("io_uring_enter")
        self.pending -= int(rc)
        self.inflight += int(rc)
        self._reap()
        return int(rc)

    def _reap(self) -> None:
        head = _u32_at(self._cq_mm, self._cq_head_off)
        tail = fstcpp.ring_load_acquire(self._cq_base + self._cq_tail_off)
        while head != tail:
            off = self._cq_cqes_off + (head & self._cq_mask) * _CQE_SIZE
            user_data, res, _flags = struct.unpack_from("=QiI", self._cq_mm, off)
            self._completions[user_data] = res
            self.inflight -= 1
            head = (head + 1) & 0xFFFFFFFF
        fstcpp.ring_store_release(self._cq_base + self._cq_head_off, head)

    def wait(self, user_data: int) -> int:
        """Block until the read ``user_data`` completed and return its result."""
        while user_data not in self._completions:
            if self.pending:
                self.submit(wait_nr=1)
                continue
            if self.inflight == 0:
                raise RuntimeError(f"request {user_data} was never submitted")
            rc = _syscall(
                _SYS_IO_URING_ENTER,
                ctypes.c_uint(self.fd),
                ctypes.c_uint(0),
                ctypes.c_uint(1),
                ctypes.c_uint(IORING_ENTER_GETEVENTS),
                None,
                ctypes.c_size_t(0),
            )
            if rc < 0:
                err = ctypes.get_errno()
                if err == errno.EINTR:
                    continue
                raise OSError(err, f"io_uring_enter(wait): {os.strerror(err)}")
            self._reap()
        return self._completions.pop(user_data)

    # -- teardown -----------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._sq_anchor = self._cq_anchor = None
        for mm in {
            id(self._sqes): self._sqes,
            id(self._sq_mm): self._sq_mm,
            id(self._cq_mm): self._cq_mm,
        }.values():
            try:
                mm.close()
            except Exception:
                pass
        os.close(self.fd)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# DMA-BUF export of device memory
# ---------------------------------------------------------------------------


def export_device_range(ptr: int, size: int, is_hip: bool) -> Tuple[int, int, int]:
    """Export ``[ptr, ptr+size)`` of GPU memory as a DMA-BUF.

    Returns ``(fd, offset, dmabuf_size)``. ``offset`` is where ``ptr`` sits
    inside the exported buffer: NVIDIA exports exactly the requested range
    (offset 0); ROCm may export the whole backing allocation and reports where
    the slice starts. ``dmabuf_size`` is the exported buffer's full size, which
    is what the kernel's registration cap applies to.
    """
    if ptr % _PAGE or size % _PAGE:
        raise ValueError(
            f"DMA-BUF export needs a page-aligned range, got {ptr:#x}+{size}"
        )
    if is_hip:
        lib = ctypes.CDLL("libhsa-runtime64.so.1")
        fn = lib.hsa_amd_portable_export_dmabuf
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        fn.restype = ctypes.c_int
        fd = ctypes.c_int(-1)
        off = ctypes.c_uint64(0)
        rc = fn(ptr, size, ctypes.byref(fd), ctypes.byref(off))
        if rc != 0:
            raise RuntimeError(f"hsa_amd_portable_export_dmabuf failed: status {rc}")
        offset = int(off.value)
    else:
        lib = _cuda_driver()
        fn = lib.cuMemGetHandleForAddressRange
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_uint64,
        ]
        fn.restype = ctypes.c_int
        fd = ctypes.c_int(-1)
        CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 1
        rc = fn(ctypes.byref(fd), ptr, size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0)
        if rc != 0:
            raise RuntimeError(
                f"cuMemGetHandleForAddressRange failed: CUresult {rc} "
                "(needs the NVIDIA open kernel modules)"
            )
        offset = 0
    dmabuf_size = os.fstat(fd.value).st_size
    if offset + size > dmabuf_size:
        os.close(fd.value)
        raise RuntimeError(
            f"exported DMA-BUF of {dmabuf_size} bytes does not cover "
            f"offset {offset} + {size}"
        )
    return fd.value, offset, dmabuf_size


class RegisteredRegion:
    """One registered slot and the byte range of the destination it covers."""

    __slots__ = ("slot", "dev_start", "dev_end", "buf_off", "fd")

    def __init__(self, slot: int, dev_start: int, dev_end: int, buf_off: int, fd: int):
        self.slot = slot
        self.dev_start = dev_start  # absolute destination address
        self.dev_end = dev_end
        self.buf_off = buf_off  # READ_FIXED addr for dev_start
        self.fd = fd  # dma-buf fd, or -1 for host memory


def register_destination(
    ring: IoUring,
    base: int,
    length: int,
    target_fd: int,
    is_device: bool,
    is_hip: bool,
    max_bytes: int = DMABUF_REGISTER_MAX_BYTES,
) -> List[RegisteredRegion]:
    """Register ``[base, base+length)`` for reads from ``target_fd``.

    Device memory is exported and registered in slices of at most
    ``max_bytes`` so each DMA-BUF stays under the kernel's per-registration
    cap. Host memory is registered as one ordinary fixed buffer. On failure,
    every slice registered so far is released.
    """
    if base % _PAGE or length % _PAGE:
        raise ValueError(f"destination {base:#x}+{length} is not page aligned")
    regions: List[RegisteredRegion] = []
    try:
        if not is_device:
            slot = ring.register_host(base, length)
            regions.append(RegisteredRegion(slot, base, base + length, base, -1))
            return regions
        off = 0
        while off < length:
            chunk = min(max_bytes, length - off)
            fd, export_off, dmabuf_size = export_device_range(base + off, chunk, is_hip)
            try:
                slot = ring.register_dmabuf(fd, target_fd)
            except BaseException:
                os.close(fd)
                raise
            regions.append(
                RegisteredRegion(slot, base + off, base + off + chunk, export_off, fd)
            )
            off += chunk
    except BaseException:
        release_regions(ring, regions)
        raise
    return regions


def release_regions(
    ring: IoUring, regions: List[RegisteredRegion]
) -> Optional[Exception]:
    """Unregister and close every region; return the first error, if any."""
    first: Optional[Exception] = None
    for region in regions:
        try:
            ring.unregister(region.slot)
        except Exception as exc:
            if first is None:
                first = exc
        if region.fd >= 0:
            try:
                os.close(region.fd)
            except OSError as exc:
                if first is None:
                    first = exc
            region.fd = -1
    regions.clear()
    return first


# ---------------------------------------------------------------------------
# Device runtime: HIP runtime on ROCm, CUDA driver API on NVIDIA
# ---------------------------------------------------------------------------
#
# On ROCm the copier calls the HIP runtime, libamdhip64.so, which resolves to
# the library the framework already loaded.
#
# On NVIDIA it calls the CUDA driver API in libcuda.so.1, not the runtime.
# Every NVIDIA driver installs libcuda.so.1 under that name, and there is
# one driver per process. The runtime is different: pip wheels ship only
# versioned names (libcudart.so.12, libcudart.so.13), so opening a bare
# libcudart.so fails in a pip-only environment, and where a CUDA toolkit is
# installed it can open a second runtime, of another version, next to the
# framework's. The driver API works in the device's primary context, which
# is the context the CUDA runtime, and therefore PyTorch, uses.


def _runtime(is_hip: bool):
    """The HIP runtime. NVIDIA goes through ``_cuda_driver`` instead."""
    assert is_hip, "the CUDA path uses the driver API"
    return ctypes.CDLL("libamdhip64.so")


_CUDA_DRIVER_LIB = "libcuda.so.1"
# CUdevice_attribute value from cuda.h.
CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124

_CUDA_SIGNATURES = {
    "cuInit": [ctypes.c_uint],
    "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
    "cuDeviceGetAttribute": [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int],
    "cuDevicePrimaryCtxRetain": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
    "cuCtxGetCurrent": [ctypes.POINTER(ctypes.c_void_p)],
    "cuCtxSetCurrent": [ctypes.c_void_p],
    "cuCtxGetDevice": [ctypes.POINTER(ctypes.c_int)],
    "cuCtxSynchronize": [],
    "cuMemAlloc_v2": [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t],
    "cuMemFree_v2": [ctypes.c_uint64],
    "cuMemcpyDtoD_v2": [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t],
}


def _cuda_check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed: CUresult {rc}")


@functools.lru_cache(maxsize=None)
def _cuda_driver() -> ctypes.CDLL:
    """Open and initialize the CUDA driver library once per process."""
    lib = ctypes.CDLL(_CUDA_DRIVER_LIB)
    for name, argtypes in _CUDA_SIGNATURES.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
    _cuda_check(lib.cuInit(0), "cuInit")
    return lib


@functools.lru_cache(maxsize=None)
def _cuda_primary_context(index: int) -> int:
    """Retain device ``index``'s primary context for the life of the process.

    The CUDA runtime retains a primary context the same way once it
    initializes a device, and never releases it before exit. Releasing it
    here could destroy the context, and every allocation in it, when the
    framework has not touched the device yet.
    """
    lib = _cuda_driver()
    dev = ctypes.c_int(0)
    _cuda_check(lib.cuDeviceGet(ctypes.byref(dev), index), f"cuDeviceGet({index})")
    ctx = ctypes.c_void_p(0)
    _cuda_check(
        lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev),
        f"cuDevicePrimaryCtxRetain({index})",
    )
    if not ctx.value:
        raise RuntimeError(f"cuDevicePrimaryCtxRetain({index}) returned no context")
    return int(ctx.value)


@contextlib.contextmanager
def _cuda_context(index: Optional[int]) -> Iterator[int]:
    """Make device ``index``'s primary context current; yield the device.

    With ``index`` None, keep the thread's current context, or use device
    0's primary context when the thread has none, which is the device the
    CUDA runtime picks for a thread that never selected one. The previous
    context is restored on exit.
    """
    lib = _cuda_driver()
    prev = ctypes.c_void_p(0)
    _cuda_check(lib.cuCtxGetCurrent(ctypes.byref(prev)), "cuCtxGetCurrent")
    if index is None:
        if prev.value:
            dev = ctypes.c_int(0)
            _cuda_check(lib.cuCtxGetDevice(ctypes.byref(dev)), "cuCtxGetDevice")
            yield dev.value
            return
        index = 0
    target = _cuda_primary_context(index)
    if prev.value == target:
        yield index
        return
    _cuda_check(lib.cuCtxSetCurrent(target), f"cuCtxSetCurrent({index})")
    try:
        yield index
    finally:
        lib.cuCtxSetCurrent(prev)


def cuda_dmabuf_supported(index: Optional[int]) -> bool:
    """Whether the driver can export device ``index``'s memory as a DMA-BUF.

    GeForce drivers report 0 and refuse ``cuMemGetHandleForAddressRange``
    with ``CUDA_ERROR_NOT_SUPPORTED`` (801). Raises ``OSError`` when the
    driver library cannot be opened and ``RuntimeError`` when a driver call
    fails.
    """
    lib = _cuda_driver()
    with _cuda_context(index) as dev_index:
        dev = ctypes.c_int(0)
        _cuda_check(lib.cuDeviceGet(ctypes.byref(dev), dev_index), "cuDeviceGet")
        value = ctypes.c_int(0)
        _cuda_check(
            lib.cuDeviceGetAttribute(
                ctypes.byref(value), CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev
            ),
            "cuDeviceGetAttribute(DMA_BUF_SUPPORTED)",
        )
    return value.value != 0


@contextlib.contextmanager
def device_scope(index: Optional[int], is_hip: bool) -> Iterator[None]:
    """Make ``index`` the calling thread's current device, then restore it.

    The current device is per thread and starts at 0 in a new thread, such
    as a pipelined loader's producer. Allocation, export and device copies
    for a load must run on the target device. With ``index`` None, ROCm
    keeps the thread's current device; on NVIDIA the thread's current
    context is kept, or device 0's primary context is made current for the
    scope when the thread has none.
    """
    if not is_hip:
        with _cuda_context(index):
            yield
        return
    if index is None:
        yield
        return
    lib = _runtime(is_hip)
    get = lib.hipGetDevice
    set_ = lib.hipSetDevice
    get.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get.restype = ctypes.c_int
    set_.argtypes = [ctypes.c_int]
    set_.restype = ctypes.c_int
    prev = ctypes.c_int(0)
    rc = get(ctypes.byref(prev))
    if rc != 0:
        raise RuntimeError(f"hipGetDevice: {rc}")
    if prev.value == index:
        yield
        return
    rc = set_(index)
    if rc != 0:
        raise RuntimeError(f"hipSetDevice({index}): {rc}")
    try:
        yield
    finally:
        set_(prev.value)


# ---------------------------------------------------------------------------
# Dedicated device allocations (one driver allocation per buffer)
# ---------------------------------------------------------------------------
#
# A caching allocator may hand out a slice of a much larger driver allocation,
# and the ROCm exporter exports the whole allocation that contains a pointer.
# The kernel limits the size of each registered DMA-BUF, so an arena piece
# that must be registered needs its own driver allocation of at most the
# copier's registration size. These helpers allocate straight from the
# device runtime and record the pointer so the framework frees it the same
# way.

# ptr -> (size, is_hip, CUDA device index or -1)
_DEDICATED: Dict[int, Tuple[int, bool, int]] = {}
_HIP_MEMCPY_D2D = 3


def dedicated_malloc(size: int, is_hip: bool) -> int:
    """Allocate ``size`` bytes on the current device.

    Call it inside ``device_scope`` for the target device.
    """
    if not is_hip:
        lib = _cuda_driver()
        with _cuda_context(None) as dev_index:
            dptr = ctypes.c_uint64(0)
            rc = lib.cuMemAlloc_v2(ctypes.byref(dptr), size)
        if rc != 0 or not dptr.value:
            raise MemoryError(f"cuMemAlloc({size}) failed: CUresult {rc}")
        _DEDICATED[int(dptr.value)] = (size, False, dev_index)
        return int(dptr.value)
    lib = _runtime(is_hip)
    fn = lib.hipMalloc
    fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    fn.restype = ctypes.c_int
    ptr = ctypes.c_void_p(0)
    rc = fn(ctypes.byref(ptr), size)
    if rc != 0 or not ptr.value:
        raise MemoryError(f"hipMalloc({size}) failed: {rc}")
    _DEDICATED[int(ptr.value)] = (size, True, -1)
    return int(ptr.value)


def release_dedicated(ptr: int) -> bool:
    """Free a dedicated allocation; False when ``ptr`` is not one."""
    entry = _DEDICATED.pop(ptr, None)
    if entry is None:
        return False
    _size, is_hip, dev_index = entry
    if not is_hip:
        lib = _cuda_driver()
        # The caller may be any thread, so free in the allocation's own
        # context. cudaFree waits for the device first; keep that, so no
        # queued copy still uses the memory.
        with _cuda_context(dev_index):
            _cuda_check(lib.cuCtxSynchronize(), "cuCtxSynchronize")
            _cuda_check(lib.cuMemFree_v2(ptr), "cuMemFree")
        return True
    lib = _runtime(is_hip)
    fn = lib.hipFree
    fn.argtypes = [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    rc = fn(ptr)
    if rc != 0:
        raise RuntimeError(f"hipFree failed: {rc}")
    return True


def device_memcpy(dst: int, src: int, size: int, is_hip: bool) -> None:
    """Device-to-device copy, ordered on the legacy default stream.

    As with ``cudaMemcpy``, the copy is ordered with later work on the same
    device but the host does not wait for it.
    """
    if not is_hip:
        lib = _cuda_driver()
        with _cuda_context(None):
            _cuda_check(lib.cuMemcpyDtoD_v2(dst, src, size), "cuMemcpyDtoD")
        return
    lib = _runtime(is_hip)
    fn = lib.hipMemcpy
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    fn.restype = ctypes.c_int
    rc = fn(dst, src, size, _HIP_MEMCPY_D2D)
    if rc != 0:
        raise RuntimeError(f"hipMemcpy failed: {rc}")
