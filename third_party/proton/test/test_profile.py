import torch
import triton
import triton.profiler as proton
import tempfile
import json
import pytest
from typing import NamedTuple

import triton.language as tl


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@pytest.mark.parametrize("context", ["shadow", "python"])
def test_torch(context):
    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        proton.start(f.name.split(".")[0], context=context)
        proton.enter_scope("test")
        torch.ones((2, 2), device="cuda")
        proton.exit_scope()
        proton.finalize()
        data = json.load(f)
        if context == "shadow":
            assert len(data[0]["children"]) == 1
            assert data[0]["children"][0]["frame"]["name"] == "test"
            assert data[0]["children"][0]["children"][0]["metrics"]["Time (ns)"] > 0
        elif context == "python":
            assert len(data[0]["children"]) == 1
            # The last frame is the torch kernel
            prev_frame = data
            curr_frame = data[0]["children"]
            while len(curr_frame) > 0:
                prev_frame = curr_frame
                curr_frame = curr_frame[0]["children"]
            assert "elementwise_kernel" in prev_frame[0]["frame"]["name"]


def test_triton():

    @triton.jit
    def foo(x, y):
        tl.store(y, tl.load(x))

    x = torch.tensor([2], device="cuda")
    y = torch.zeros_like(x)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        proton.start(f.name.split(".")[0])
        with proton.scope("test0"):
            with proton.scope("test1"):
                foo[(1, )](x, y)
        with proton.scope("test2"):
            foo[(1, )](x, y)
        proton.finalize()
        data = json.load(f)
        assert len(data[0]["children"]) == 2
        assert data[0]["children"][0]["frame"]["name"] == "test0"
        assert len(data[0]["children"][0]["children"]) == 1
        assert data[0]["children"][0]["children"][0]["frame"]["name"] == "test1"
        assert data[0]["children"][1]["frame"]["name"] == "test2"


def test_cudagraph():
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    @triton.jit
    def foo(x, y, z):
        tl.store(z, tl.load(y) + tl.load(x))

    def fn():
        a = torch.ones((2, 2), device="cuda")
        b = torch.ones((2, 2), device="cuda")
        c = a + b
        foo[(1, )](a, b, c)

    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        proton.start(f.name.split(".")[0], context="shadow")

        # warmup
        # four kernels
        fn()

        # no kernels
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(10):
                fn()

        proton.enter_scope("test")
        g.replay()
        g.reset()
        torch.cuda.synchronize()
        proton.exit_scope()
        proton.finalize()

        data = json.load(f)
        # CUDA/HIP graph may also invoke additional kernels to reset outputs
        # {torch.ones, add, foo, test}
        assert len(data[0]["children"]) >= 4
        # find the test frame
        test_frame = None
        for child in data[0]["children"]:
            if child["frame"]["name"] == "test":
                test_frame = child
                break
        assert test_frame is not None
        # {torch.ones, add, foo}
        if is_hip():
            assert len(test_frame["children"]) >= 2
        else:
            assert len(test_frame["children"]) >= 3
        assert test_frame["children"][0]["metrics"]["Time (ns)"] > 0


def test_metrics():

    @triton.jit
    def foo(x, y):
        tl.store(y, tl.load(x))

    x = torch.tensor([2], device="cuda")
    y = torch.zeros_like(x)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        proton.start(f.name.split(".")[0])
        with proton.scope("test0", {"foo": 1.0}):
            foo[(1, )](x, y)
        proton.finalize()
        data = json.load(f)
        assert len(data[0]["children"]) == 1
        assert data[0]["children"][0]["frame"]["name"] == "test0"
        assert data[0]["children"][0]["metrics"]["foo"] == 1.0


def test_metrics_ignore():

    @triton.jit
    def foo(x, y):
        tl.store(y, tl.load(x))

    x = torch.tensor([2], device="cuda")
    y = torch.zeros_like(x)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        session_id = proton.start(f.name.split(".")[0])
        proton.deactivate(session_id)
        with proton.scope("test0", {"foo": 1.0}):
            foo[(1, )](x, y)
        proton.activate(session_id)
        proton.finalize()
        data = json.load(f)
        assert len(data[0]["children"]) == 0


def test_scope_backward():
    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        proton.start(f.name.split(".")[0])
        with proton.scope("ones1"):
            a = torch.ones((100, 100), device="cuda", requires_grad=True)
        with proton.scope("plus"):
            a2 = a * a * a
        with proton.scope("ones2"):
            loss = torch.ones_like(a2)

        # Backward triggers two kernels in a single scope
        with proton.scope("backward"):
            a2.backward(loss)
        proton.finalize()
        data = json.load(f)
        assert len(data[0]["children"]) == 4


def test_hook():

    def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        # get arg's element size
        element_size = args["x"].element_size()  # non-const
        size = args["size"]  # const
        key = "flops" + str(element_size * 8)
        num_ctas = metadata.num_ctas
        return {"name": f"foo_test_{num_ctas}ctas_{size}elems", key: 1.0}

    @triton.jit(launch_metadata=metadata_fn)
    def foo(x, size: tl.constexpr, y):
        offs = tl.arange(0, size)
        tl.store(y + offs, tl.load(x + offs))

    x = torch.tensor([2], device="cuda", dtype=torch.float32)
    y = torch.zeros_like(x)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        proton.start(f.name.split(".")[0], hook="triton")
        with proton.scope("test0"):
            foo[(1, )](x, 1, y, num_warps=4)
        proton.finalize()
        data = json.load(f)
        assert len(data[0]["children"]) == 1
        assert data[0]["children"][0]["frame"]["name"] == "test0"
        assert data[0]["children"][0]["children"][0]["frame"]["name"] == "foo_test_1ctas_1elems"
        assert data[0]["children"][0]["children"][0]["metrics"]["flops32"] == 1.0
        assert data[0]["children"][0]["children"][0]["metrics"]["Time (ns)"] > 0
