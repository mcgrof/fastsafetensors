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


@pytest.mark.parametrize("is_hip", [True, False])
def test_device_scope_selects_and_restores(monkeypatch, is_hip):
    import fastsafetensors.io_uring_dmabuf as iod

    fake = _FakeRuntime(current=0)
    monkeypatch.setattr(iod, "_runtime", lambda hip: fake)
    with iod.device_scope(3, is_hip):
        assert fake.current == 3
    assert fake.current == 0 and fake.calls == [3, 0]
    fake.calls.clear()
    with iod.device_scope(0, is_hip):
        pass
    assert fake.calls == []


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
