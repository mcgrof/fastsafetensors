# SPDX-License-Identifier: Apache-2.0

"""Exercise the io_uring layer the dmabuf copier is built on."""

import ctypes
import errno
import fcntl
import mmap
import os
import struct

import pytest

from fastsafetensors.io_uring_dmabuf import IoUring


def test_ring_index_helpers_round_trip():
    """The ordered ring accesses read and write the addressed 32-bit word."""
    from fastsafetensors import cpp

    words = (ctypes.c_uint32 * 2)(0xDEADBEEF, 0)
    base = ctypes.addressof(words)
    assert cpp.ring_load_acquire(base) == 0xDEADBEEF
    cpp.ring_store_release(base + 4, 0xFFFFFFFF)
    assert words[1] == 0xFFFFFFFF and words[0] == 0xDEADBEEF


class _FakeRuntime:
    """Records the runtime's current-device calls for one thread."""

    def __init__(self, current):
        self.current = current
        self.calls = []

        def get(ptr):
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int))[0] = self.current
            return 0

        def set_(index):
            self.calls.append(index)
            self.current = index
            return 0

        self.hipGetDevice = self.cudaGetDevice = _Fn(get)
        self.hipSetDevice = self.cudaSetDevice = _Fn(set_)


class _Fn:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


def test_device_scope_selects_and_restores_hip(monkeypatch):
    import fastsafetensors.io_uring_dmabuf as iod

    fake = _FakeRuntime(current=0)
    monkeypatch.setattr(iod, "_runtime", lambda hip: fake)
    with iod.device_scope(3, True):
        assert fake.current == 3
    assert fake.current == 0 and fake.calls == [3, 0]
    fake.calls.clear()
    with iod.device_scope(0, True):
        pass
    with iod.device_scope(None, True):
        pass
    assert fake.calls == []


class _FakeDriver:
    """Models the CUDA driver calls the copier makes, for one thread.

    Context handles are 0x1000 + device index; 0 means no current context.
    """

    def __init__(self, current=0, dma_buf_supported=1):
        self.current = current
        self.sets = []
        self.allocs = {}
        self.events = []
        self.next_ptr = 0x7000_0000
        self.dma_buf_supported = dma_buf_supported

        def out(ptr, ctype, value):
            ctypes.cast(ptr, ctypes.POINTER(ctype))[0] = value

        def get_current(p):
            out(p, ctypes.c_void_p, self.current or None)
            return 0

        def set_current(ctx):
            ctx = ctx.value if isinstance(ctx, ctypes.c_void_p) else ctx
            self.sets.append(ctx or 0)
            self.current = ctx or 0
            return 0

        def get_device(p):
            if not self.current:
                return 201  # CUDA_ERROR_INVALID_CONTEXT
            out(p, ctypes.c_int, self.current - 0x1000)
            return 0

        def device_get(p, index):
            out(p, ctypes.c_int, index)
            return 0

        def attribute(p, attr, dev):
            out(p, ctypes.c_int, self.dma_buf_supported if attr == 124 else 0)
            return 0

        def alloc(p, size):
            if not self.current:
                return 201
            ptr = self.next_ptr
            self.next_ptr += 1 << 20
            self.allocs[ptr] = self.current
            out(p, ctypes.c_uint64, ptr)
            return 0

        def sync():
            self.events.append(("sync", self.current))
            return 0

        def free(ptr):
            if self.allocs.pop(ptr, None) != self.current:
                return 201  # freed outside its own context
            self.events.append(("free", self.current))
            return 0

        def copy(dst, src, size):
            if not self.current:
                return 201
            self.events.append(("copy", self.current))
            return 0

        self.cuCtxGetCurrent = _Fn(get_current)
        self.cuCtxSetCurrent = _Fn(set_current)
        self.cuCtxGetDevice = _Fn(get_device)
        self.cuDeviceGet = _Fn(device_get)
        self.cuDeviceGetAttribute = _Fn(attribute)
        self.cuMemAlloc_v2 = _Fn(alloc)
        self.cuCtxSynchronize = _Fn(sync)
        self.cuMemFree_v2 = _Fn(free)
        self.cuMemcpyDtoD_v2 = _Fn(copy)


@pytest.fixture
def fake_driver(monkeypatch):
    import fastsafetensors.io_uring_dmabuf as iod

    def install(**kwargs):
        fake = _FakeDriver(**kwargs)
        monkeypatch.setattr(iod, "_cuda_driver", lambda: fake)
        monkeypatch.setattr(iod, "_cuda_primary_context", lambda i: 0x1000 + i)
        monkeypatch.setattr(
            iod, "_runtime", lambda hip: pytest.fail("CUDA must not load a runtime")
        )
        return fake

    return install


def test_device_scope_selects_and_restores_cuda(fake_driver):
    import fastsafetensors.io_uring_dmabuf as iod

    fake = fake_driver(current=0x1000)
    with iod.device_scope(3, False):
        assert fake.current == 0x1003
    assert fake.current == 0x1000 and fake.sets == [0x1003, 0x1000]
    fake.sets.clear()
    with iod.device_scope(0, False):
        pass
    with iod.device_scope(None, False):
        pass
    assert fake.sets == []


def test_cuda_thread_without_context_uses_device_zero(fake_driver):
    """A new thread has no current context; the runtime would pick device 0."""
    import fastsafetensors.io_uring_dmabuf as iod

    fake = fake_driver(current=0)
    with iod.device_scope(None, False):
        assert fake.current == 0x1000
    assert fake.current == 0
    ptr = iod.dedicated_malloc(4096, False)
    assert fake.allocs[ptr] == 0x1000 and fake.current == 0
    iod.device_memcpy(ptr, ptr, 4096, False)
    assert fake.events == [("copy", 0x1000)] and fake.current == 0
    assert iod.release_dedicated(ptr)


def test_cuda_free_runs_in_the_allocation_context(fake_driver):
    """The framework may free from another thread or device."""
    import fastsafetensors.io_uring_dmabuf as iod

    fake = fake_driver(current=0x1000)
    with iod.device_scope(2, False):
        ptr = iod.dedicated_malloc(4096, False)
    assert fake.allocs[ptr] == 0x1002
    assert iod.release_dedicated(ptr)
    assert fake.events == [("sync", 0x1002), ("free", 0x1002)]
    assert fake.current == 0x1000
    assert not iod.release_dedicated(ptr)


@pytest.mark.parametrize("value", [0, 1])
def test_cuda_dmabuf_supported_reads_the_attribute(fake_driver, value):
    import fastsafetensors.io_uring_dmabuf as iod

    fake_driver(current=0, dma_buf_supported=value)
    assert iod.cuda_dmabuf_supported(1) is bool(value)
    assert iod.cuda_dmabuf_supported(None) is bool(value)


_UDMABUF_CREATE = (1 << 30) | (24 << 16) | (ord("u") << 8) | 0x42
_UDMABUF_FLAGS_CLOEXEC = 0x01
# Linux UAPI values, for Python builds whose os/fcntl modules lack them.
_MFD_ALLOW_SEALING = getattr(os, "MFD_ALLOW_SEALING", 0x0002)
_F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
_F_SEAL_SHRINK = getattr(fcntl, "F_SEAL_SHRINK", 0x0002)


def _memfd(name):
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, _MFD_ALLOW_SEALING)
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.memfd_create(name.encode(), ctypes.c_uint(_MFD_ALLOW_SEALING))
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return fd


def _udmabuf(size):
    """A DMA-BUF over a sealed memfd, and a mapping of that memfd."""
    memfd = _memfd("fst-udmabuf")
    os.ftruncate(memfd, size)
    fcntl.fcntl(memfd, _F_ADD_SEALS, _F_SEAL_SHRINK)
    dev = os.open("/dev/udmabuf", os.O_RDWR | os.O_CLOEXEC)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        arg = ctypes.create_string_buffer(
            struct.pack("IIQQ", memfd, _UDMABUF_FLAGS_CLOEXEC, 0, size)
        )
        fd = libc.ioctl(dev, ctypes.c_ulong(_UDMABUF_CREATE), arg)
        if fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
    finally:
        os.close(dev)
    view = mmap.mmap(memfd, size)
    os.close(memfd)
    return fd, view


def test_read_fixed_into_registered_dmabuf(tmp_dir):
    """Register a real DMA-BUF and read a file into it with READ_FIXED.

    Runs where the kernel has io_uring DMA-BUF registered buffers for the
    test directory's file system and /dev/udmabuf is accessible; skips
    elsewhere. The other tests register host memory instead.
    """
    if not os.access("/dev/udmabuf", os.R_OK | os.W_OK):
        pytest.skip("/dev/udmabuf is not accessible")
    size = 1 << 20
    path = os.path.join(tmp_dir, "udmabuf-source.bin")
    payload = os.urandom(size)
    with open(path, "wb") as f:
        f.write(payload)
    try:
        target = os.open(path, os.O_RDONLY | os.O_DIRECT)
    except OSError:
        pytest.skip("test directory rejects O_DIRECT")
    try:
        ring = IoUring(entries=4, buffer_slots=2)
    except OSError as exc:
        os.close(target)
        pytest.skip(f"io_uring unavailable: {exc}")
    dmabuf, view = _udmabuf(size)
    try:
        try:
            slot = ring.register_dmabuf(dmabuf, target)
        except OSError as exc:
            pytest.skip(f"kernel refused DMA-BUF registration: {exc}")
        user_data = ring.read_fixed(target, 0, slot, 0, size)
        ring.submit()
        res = ring.wait(user_data)
        ring.unregister(slot)
        if res in (-errno.EOPNOTSUPP, -errno.EINVAL):
            pytest.skip(f"file system has no DMA-BUF direct I/O: {res}")
        assert res == size
        assert view[:] == payload
    finally:
        ring.close()
        view.close()
        os.close(dmabuf)
        os.close(target)
