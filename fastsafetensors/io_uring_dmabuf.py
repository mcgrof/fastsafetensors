# SPDX-License-Identifier: Apache-2.0
"""Minimal io_uring submission layer for DMA-BUF and host fixed buffers.

The dmabuf copier needs three things from the kernel: a ring, a fixed-buffer
table whose slots can hold a DMA-BUF bound to one target file, and
``IORING_OP_READ_FIXED`` submissions against those slots. liburing does not
expose the DMA-BUF registration descriptor yet, so this module talks to the
raw system calls through ``ctypes``. It deliberately covers nothing else.

Registration ABI (``io_uring_register(IORING_REGISTER_BUFFERS_UPDATE)`` with
``IORING_RSRC_UPDATE_EXTENDED`` and ``IO_REGBUF_TYPE_DMABUF``) follows the
``dmabuf-size-ceiling`` kernel series: the descriptor carries the DMA-BUF fd
and the file descriptor whose device will DMA into it; ``uaddr`` and ``size``
must be zero. A ``READ_FIXED`` into such a slot addresses the buffer by byte
offset from the start of the DMA-BUF, not by a user pointer.

Host memory is registered through the ordinary ``iovec`` update so the same
ring and the same copier run on a CPU-only host, where the tests live.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import struct
from typing import Dict, List, Optional, Tuple

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
IORING_REGISTER_IOWQ_MAX_WORKERS = 19

# Every DMA-BUF io_uring kernel registers at least this many bytes per
# DMA-BUF; early versions of the series refuse anything larger.
DMABUF_REGISTER_MAX_BYTES = 1 << 30

# The merged series refuses a DMA-BUF larger than UINT_MAX bytes, because the
# offset into it travels in a 32-bit bvec field. The ROCm exporter exports the
# whole driver allocation, which it rounds to 2 MiB, so the largest allocation
# that stays under the limit is 4 GiB - 2 MiB.
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
    return struct.unpack_from("<I", buf, off)[0]


def _u32_set(buf: mmap.mmap, off: int, value: int) -> None:
    struct.pack_into("<I", buf, off, value & 0xFFFFFFFF)


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

    def set_max_workers(self, bounded: int, unbounded: int = 0) -> Tuple[int, int]:
        """Cap the io-wq worker pools; zero leaves a pool unchanged.

        DMA-BUF reads are punted to bounded io-wq workers, whose default cap
        is four, so this bounds the number of reads the kernel executes at
        once. Returns the previous caps.
        """
        arr = (ctypes.c_uint * 2)(bounded, unbounded)
        self._register(IORING_REGISTER_IOWQ_MAX_WORKERS, arr, 2)
        return int(arr[0]), int(arr[1])

    def _take_slot(self) -> int:
        if not self._free_slots:
            raise RuntimeError("io_uring fixed-buffer table is full")
        return self._free_slots.pop(0)

    def register_dmabuf(self, dmabuf_fd: int, target_fd: int) -> int:
        """Bind a DMA-BUF to ``target_fd`` in a free slot and return the slot.

        The kernel builds the DMA mapping lazily on the first read, so a
        registration that succeeds here can still fail at I/O time with
        ``EOPNOTSUPP`` on a file system or device without DMA-BUF support.
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
        head = _u32_at(self._sq_mm, self._sq_head_off)
        if tail - head >= self._sq_entries:
            raise RuntimeError("submission queue is full; call submit() first")
        idx = tail & self._sq_mask
        user_data = self._next_user_data
        self._next_user_data += 1
        sqe = struct.pack(
            "<BBHiQQIIQHHiQQ",
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
        _u32_set(self._sq_mm, self._sq_tail_off, tail + 1)
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
        tail = _u32_at(self._cq_mm, self._cq_tail_off)
        while head != tail:
            off = self._cq_cqes_off + (head & self._cq_mask) * _CQE_SIZE
            user_data, res, _flags = struct.unpack_from("<QiI", self._cq_mm, off)
            self._completions[user_data] = res
            self.inflight -= 1
            head += 1
        _u32_set(self._cq_mm, self._cq_head_off, head)

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
                if err in (4,):  # EINTR
                    continue
                raise OSError(err, f"io_uring_enter(wait): {os.strerror(err)}")
            self._reap()
        return self._completions.pop(user_data)

    # -- teardown -----------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
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
        lib = ctypes.CDLL("libcuda.so.1")
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
# Dedicated device allocations (one driver allocation per buffer)
# ---------------------------------------------------------------------------
#
# A caching allocator may hand out a slice of a much larger driver allocation,
# and the ROCm exporter exports the whole allocation that contains a pointer.
# The kernel limits the size of each registered DMA-BUF, so an arena piece
# that must be registered needs its own driver allocation of at most the
# copier's registration size. These helpers allocate straight from the runtime and record
# the pointer so the framework frees it through the runtime as well.

_DEDICATED: Dict[int, Tuple[int, bool]] = {}  # ptr -> (size, is_hip)
_HIP_MEMCPY_D2D = 3
_CUDA_MEMCPY_D2D = 3


def _runtime(is_hip: bool):
    return ctypes.CDLL("libamdhip64.so" if is_hip else "libcudart.so")


def dedicated_malloc(size: int, is_hip: bool) -> int:
    """Allocate ``size`` bytes on the current device with the runtime."""
    lib = _runtime(is_hip)
    fn = lib.hipMalloc if is_hip else lib.cudaMalloc
    fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    fn.restype = ctypes.c_int
    ptr = ctypes.c_void_p(0)
    rc = fn(ctypes.byref(ptr), size)
    if rc != 0 or not ptr.value:
        raise MemoryError(
            f"{'hipMalloc' if is_hip else 'cudaMalloc'}({size}) failed: {rc}"
        )
    _DEDICATED[int(ptr.value)] = (size, is_hip)
    return int(ptr.value)


def release_dedicated(ptr: int) -> bool:
    """Free a dedicated allocation; False when ``ptr`` is not one."""
    entry = _DEDICATED.pop(ptr, None)
    if entry is None:
        return False
    _size, is_hip = entry
    lib = _runtime(is_hip)
    fn = lib.hipFree if is_hip else lib.cudaFree
    fn.argtypes = [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    rc = fn(ptr)
    if rc != 0:
        raise RuntimeError(f"{'hipFree' if is_hip else 'cudaFree'} failed: {rc}")
    return True


def device_memcpy(dst: int, src: int, size: int, is_hip: bool) -> None:
    """Synchronous device-to-device copy through the runtime."""
    lib = _runtime(is_hip)
    fn = lib.hipMemcpy if is_hip else lib.cudaMemcpy
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    fn.restype = ctypes.c_int
    rc = fn(dst, src, size, _HIP_MEMCPY_D2D if is_hip else _CUDA_MEMCPY_D2D)
    if rc != 0:
        raise RuntimeError(f"{'hipMemcpy' if is_hip else 'cudaMemcpy'} failed: {rc}")
