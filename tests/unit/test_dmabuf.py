# SPDX-License-Identifier: Apache-2.0

"""Exercise the dmabuf copier end to end on a CPU device.

On a CPU device the copier registers ordinary host memory with io_uring and
issues the same READ_FIXED submissions it would issue against a DMA-BUF, so
the ring, the alignment plan, the request accounting and the tensor views are
all real. Only the GPU export and the kernel's DMA-BUF binding are out of
reach here; those run on the GPU host.
"""

import os
from contextlib import closing

import pytest

from fastsafetensors import SafeTensorsFileLoader, SingleGroup
from fastsafetensors.copier import dmabuf as dmabuf_mod
from fastsafetensors.copier.dmabuf import DmabufFileCopier
from fastsafetensors.copier.registry import copier_class_of
from fastsafetensors.io_uring_dmabuf import IoUring
from fastsafetensors.st_types import DType


def _ring_available() -> bool:
    try:
        with closing(IoUring(entries=4, buffer_slots=2)):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _ring_available(), reason="io_uring with sparse fixed buffers unavailable"
)


def _same(framework, fb, name, expect) -> bool:
    got = fb.get_tensor(name)
    if tuple(got.shape) != tuple(expect.shape) or got.dtype != expect.dtype:
        return False
    return framework.is_equal(fb.get_tensor_wrapped(name), expect)


def _reference(framework, path):
    loader = SafeTensorsFileLoader(
        SingleGroup(), "cpu", nogds=True, framework=framework.get_name()
    )
    loader.add_filenames({0: [path]})
    fb = loader.copy_files_to_device()
    ref = {k: fb.get_tensor(k).clone() for k in fb.key_to_rank_lidx}
    fb.close()
    loader.close()
    return ref


def test_dmabuf_selects_copier(framework, fstcpp_log):
    loader = SafeTensorsFileLoader(
        SingleGroup(), "cpu", use_dmabuf=True, framework=framework.get_name()
    )
    try:
        assert copier_class_of(loader.copier_constructor) is DmabufFileCopier
    finally:
        loader.close()


@pytest.mark.parametrize("nogds", [False, True])
def test_dmabuf_selected_by_environment(framework, fstcpp_log, monkeypatch, nogds):
    monkeypatch.setenv("FASTSAFETENSORS_USE_DMABUF", "1")
    loader = SafeTensorsFileLoader(
        SingleGroup(), "cpu", nogds=nogds, framework=framework.get_name()
    )
    try:
        assert copier_class_of(loader.copier_constructor) is DmabufFileCopier
    finally:
        loader.close()


def test_environment_does_not_override_fgds(framework, monkeypatch):
    monkeypatch.setenv("FASTSAFETENSORS_USE_DMABUF", "1")
    seen = {}

    def capture(self, pg, device, copier_type, *args, **kwargs):
        seen["copier_type"] = copier_type

    base = SafeTensorsFileLoader.__mro__[1]
    monkeypatch.setattr(base, "__init__", capture)
    SafeTensorsFileLoader(
        SingleGroup(), "cpu", use_fgds=True, framework=framework.get_name()
    )
    assert seen["copier_type"] == "fgds"


def test_dmabuf_full_file_matches_nogds(framework, input_files, fstcpp_log):
    ref = _reference(framework, input_files[0])
    loader = SafeTensorsFileLoader(
        SingleGroup(), "cpu", use_dmabuf=True, framework=framework.get_name()
    )
    loader.add_filenames({0: [input_files[0]]})
    fb = loader.copy_files_to_device()
    try:
        assert set(fb.key_to_rank_lidx) == set(ref)
        for name, expect in ref.items():
            assert _same(framework, fb, name, expect), name
    finally:
        fb.close()
        loader.close()


def test_dmabuf_small_io_size_splits_requests(
    framework, input_files, monkeypatch, fstcpp_log
):
    monkeypatch.setenv("FASTSAFETENSORS_DMABUF_IO_BYTES", "4096")
    monkeypatch.setenv("FASTSAFETENSORS_DMABUF_QUEUE_DEPTH", "3")
    ref = _reference(framework, input_files[0])
    loader = SafeTensorsFileLoader(
        SingleGroup(), "cpu", use_dmabuf=True, framework=framework.get_name()
    )
    loader.add_filenames({0: [input_files[0]]})
    fb = loader.copy_files_to_device()
    try:
        for name, expect in ref.items():
            assert _same(framework, fb, name, expect), name
    finally:
        fb.close()
        loader.close()


def test_dmabuf_tensor_filter_reads_subset(framework, input_files, fstcpp_log):
    ref = _reference(framework, input_files[0])
    keep = sorted(ref)[: max(1, len(ref) // 2)]
    loader = SafeTensorsFileLoader(
        SingleGroup(), "cpu", use_dmabuf=True, framework=framework.get_name()
    )
    loader.add_filenames({0: [input_files[0]]})
    loader.set_tensor_filter(lambda n: n in keep)
    fb = loader.copy_files_to_device()
    try:
        for name in keep:
            assert _same(framework, fb, name, ref[name]), name
    finally:
        fb.close()
        loader.close()


def test_dmabuf_strict_refuses_unopenable_file(
    framework, input_files, tmp_dir, monkeypatch, fstcpp_log
):
    """A read failure must surface, not silently hand back garbage."""
    monkeypatch.setenv("FASTSAFETENSORS_DMABUF_STRICT", "1")
    monkeypatch.setenv("FASTSAFETENSORS_ODIRECT", "0")
    loader = SafeTensorsFileLoader(
        SingleGroup(), "cpu", use_dmabuf=True, framework=framework.get_name()
    )
    loader.add_filenames({0: [input_files[0]]})
    original = dmabuf_mod.register_destination

    def refuse(*args, **kwargs):
        raise OSError(95, "injected EOPNOTSUPP")

    monkeypatch.setattr(dmabuf_mod, "register_destination", refuse)
    try:
        with pytest.raises(RuntimeError, match="injected EOPNOTSUPP"):
            loader.copy_files_to_device()
    finally:
        monkeypatch.setattr(dmabuf_mod, "register_destination", original)
        loader.close()


def test_dmabuf_falls_back_to_nogds_when_not_strict(
    framework, input_files, monkeypatch, fstcpp_log
):
    monkeypatch.delenv("FASTSAFETENSORS_DMABUF_STRICT", raising=False)
    ref = _reference(framework, input_files[0])
    loader = SafeTensorsFileLoader(
        SingleGroup(), "cpu", use_dmabuf=True, framework=framework.get_name()
    )
    loader.add_filenames({0: [input_files[0]]})

    def refuse(*args, **kwargs):
        raise OSError(95, "injected EOPNOTSUPP")

    monkeypatch.setattr(dmabuf_mod, "register_destination", refuse)
    fb = loader.copy_files_to_device()
    try:
        for name, expect in ref.items():
            assert _same(framework, fb, name, expect), name
    finally:
        fb.close()
        loader.close()


def test_dmabuf_config_extension_maps_copier_type():
    out = SafeTensorsFileLoader.process_extension_config({"copier_type": "dmabuf"})
    assert out == {"nogds": False, "use_fgds": False, "use_dmabuf": True}


def _load_all(framework, path):
    loader = SafeTensorsFileLoader(
        SingleGroup(), "cpu", use_dmabuf=True, framework=framework.get_name()
    )
    loader.add_filenames({0: [path]})
    fb = loader.copy_files_to_device()
    return loader, fb


@pytest.fixture
def big_file(tmp_dir):
    """A shard whose tensors straddle a 16 KiB registration cap: two tensors
    above the cap (staged), small ones between them, and a zero-length one."""
    import torch
    from safetensors.torch import save_file

    torch.manual_seed(7)
    tensors = {
        "a.big": torch.randn(40 * 1024 // 2, dtype=torch.bfloat16),
        "b.small": torch.randn(3 * 1024 // 2, dtype=torch.bfloat16),
        "c.small": torch.randn(2 * 1024 // 2, dtype=torch.bfloat16),
        "d.empty": torch.empty(0, dtype=torch.bfloat16),
        "e.big": torch.randn(20 * 1024 // 4, dtype=torch.float32),
        "f.tail": torch.randn(1000 // 2, dtype=torch.bfloat16),
    }
    path = os.path.join(tmp_dir, "cap-fixture.safetensors")
    save_file(tensors, path)
    return path


def test_dmabuf_small_cap_splits_into_pieces(
    framework, big_file, monkeypatch, fstcpp_log
):
    """A tiny registration cap forces several pieces and an oversize tensor
    staged through extra registered buffers, all on the CPU path."""
    from fastsafetensors import live_allocation_count
    from fastsafetensors.common import SafeTensorsMetadata

    monkeypatch.setenv("FASTSAFETENSORS_DMABUF_REGISTER_BYTES", str(16 * 1024))
    monkeypatch.setenv("FASTSAFETENSORS_DMABUF_IO_BYTES", str(8 * 1024))
    meta = SafeTensorsMetadata.from_file(big_file, framework)
    sizes = [f.data_offsets[1] - f.data_offsets[0] for f in meta.tensors.values()]
    assert max(sizes) > 12 * 1024, "fixture must hold a tensor above the cap"
    ref = _reference(framework, big_file)
    before = live_allocation_count()
    submitted = []
    original = DmabufFileCopier._submit_piece

    def spy(self, piece):
        submitted.append((piece.oversize, len(piece.names)))
        return original(self, piece)

    monkeypatch.setattr(DmabufFileCopier, "_submit_piece", spy)
    loader, fb = _load_all(framework, big_file)
    try:
        assert len(submitted) >= 3
        assert any(oversize for oversize, _ in submitted), "no staged piece"
        assert any(not oversize for oversize, _ in submitted), "no direct piece"
        assert set(fb.key_to_rank_lidx) == set(ref)
        for name, expect in ref.items():
            assert _same(framework, fb, name, expect), name
        # every piece beyond the first is a live allocation of its own
        assert live_allocation_count() > before + 1
        keep = fb.get_tensor(sorted(ref)[0]).clone()
    finally:
        fb.close()
        loader.close()
    import gc

    gc.collect()
    assert live_allocation_count() == before
    assert keep is not None


def test_dmabuf_small_cap_with_tensor_filter(
    framework, big_file, monkeypatch, fstcpp_log
):
    monkeypatch.setenv("FASTSAFETENSORS_DMABUF_REGISTER_BYTES", str(16 * 1024))
    ref = _reference(framework, big_file)
    keep = sorted(ref)[1::2]
    loader = SafeTensorsFileLoader(
        SingleGroup(), "cpu", use_dmabuf=True, framework=framework.get_name()
    )
    loader.add_filenames({0: [big_file]})
    loader.set_tensor_filter(lambda n: n in keep)
    fb = loader.copy_files_to_device()
    try:
        for name in keep:
            assert _same(framework, fb, name, ref[name]), name
    finally:
        fb.close()
        loader.close()


def test_dmabuf_parallel_loader_chunked(framework, big_file, monkeypatch, fstcpp_log):
    """ParallelLoader sub-file chunking drives set_chunk on the copier."""
    from fastsafetensors import ParallelLoader
    from fastsafetensors.common import SafeTensorsMetadata

    monkeypatch.setenv("FASTSAFETENSORS_DMABUF_REGISTER_BYTES", str(16 * 1024))
    ref = _reference(framework, big_file)
    meta = SafeTensorsMetadata.from_file(big_file, framework)
    largest = max(f.data_offsets[1] - f.data_offsets[0] for f in meta.tensors.values())
    pl = ParallelLoader(
        pg=None,
        hf_weights_files=[big_file],
        device="cpu",
        use_dmabuf=True,
        use_tqdm_on_load=False,
        max_batch_bytes=largest + 4096,
        framework=framework.get_name(),
    )
    seen = {}
    try:
        for name, tensor in pl.iterate_weights():
            seen[name] = tensor
    finally:
        pl.close()
    assert set(seen) == set(ref)
    import torch

    for name, expect in ref.items():
        assert torch.equal(seen[name], expect), name


def test_dmabuf_many_reads_wrap_the_ring(framework, big_file, monkeypatch, fstcpp_log):
    """More reads than the ring holds at once: the submission queue wraps and
    the fixed-buffer index array is addressed past its first entries."""
    monkeypatch.setenv("FASTSAFETENSORS_DMABUF_IO_BYTES", "4096")
    monkeypatch.setenv("FASTSAFETENSORS_DMABUF_QUEUE_DEPTH", "4")
    ref = _reference(framework, big_file)
    assert os.path.getsize(big_file) > 16 * 4096
    loader, fb = _load_all(framework, big_file)
    try:
        for name, expect in ref.items():
            assert _same(framework, fb, name, expect), name
    finally:
        fb.close()
        loader.close()
