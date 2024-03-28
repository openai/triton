import functools
import os
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Dict, List
from . import language as tl


def nvsmi(attrs):
    attrs = ','.join(attrs)
    cmd = ['nvidia-smi', '-i', '0', '--query-gpu=' + attrs, '--format=csv,noheader,nounits']
    out = subprocess.check_output(cmd)
    ret = out.decode(sys.stdout.encoding).split(',')
    ret = [int(x) for x in ret]
    return ret


def do_bench_cudagraph(fn, rep=20, grad_to_none=None, return_mode="mean"):
    assert return_mode in ["min", "max", "mean", "median"]
    import torch
    """
    Benchmark the runtime of the provided function.

    :param fn: Function to benchmark
    :type fn: Callable
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    """
    if torch.cuda.current_stream() == torch.cuda.default_stream():
        raise RuntimeError("Cannot capture graph in default stream. Please use side stream in benchmark code.")
    # warmup
    fn()
    # step 1 - we estimate the amount of time the kernel call takes
    # NOTE: this estimate isn't super accurate because the GPU isn't warmed up at this point
    #       but it is probably good enough
    if grad_to_none is not None:
        for x in grad_to_none:
            x.detach_()
            x.requires_grad_(True)
            x.grad = None
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    g.replay()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event)
    n_repeat = max(1, int(rep / estimate_ms))
    # step 2 - construct a cuda graph with `n_repeat` unrolled function calls to minimize
    # host overhead
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n_repeat):
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            fn()
    torch.cuda.synchronize()
    # measure time and return
    ret = []
    n_retries = 10
    for i in range(n_retries):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        g.replay()
        end_event.record()
        torch.cuda.synchronize()
        ret += [start_event.elapsed_time(end_event) / n_repeat]
    times = torch.tensor(ret)
    return getattr(torch, return_mode)(times).item()


def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, fast_flush=True, return_mode="mean"):
    assert return_mode in ["min", "max", "mean", "median"]
    import torch
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float]
    :param fast_flush: Use faster kernel to flush L2 between measurements
    :type fast_flush: bool
    """

    fn()
    torch.cuda.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device='cuda')

    # Estimate the runtime of the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()


def assert_close(x, y, atol=None, rtol=None, err_msg=''):
    import numpy as np
    import torch

    # canonicalize arguments to be tensors
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
    # absolute tolerance
    if atol is None:
        atol = 1e-2
    atol = atol(x.dtype) if callable(atol) else atol
    # relative tolerance hook
    if rtol is None:
        rtol = 0.
    rtol = rtol(x.dtype) if callable(rtol) else rtol
    # we use numpy instead of pytorch
    # as it seems more memory efficient
    # pytorch tends to oom on large tensors
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            x = x.float()
        x = x.cpu().detach().numpy()
    if isinstance(y, torch.Tensor):
        if y.dtype == torch.bfloat16:
            y = y.float()
        y = y.cpu().detach().numpy()
    # we handle size==1 case separately as we can
    # provide better error message there
    if x.size > 1 or y.size > 1:
        np.testing.assert_allclose(x, y, atol=atol, rtol=rtol, equal_nan=True)
        return
    if not np.allclose(x, y, atol=atol, rtol=rtol):
        raise AssertionError(f'{err_msg} {x} is not close to {y} (atol={atol}, rtol={rtol})')


class Benchmark:
    """
    This class is used by the :code:`perf_report` function to generate line plots with a concise API.
    """

    def __init__(
        self,
        x_names: List[str],
        x_vals: List[Any],
        line_arg: str,
        line_vals: List[Any],
        line_names: List[str],
        plot_name: str,
        args: Dict[str, Any],
        xlabel: str = '',
        ylabel: str = '',
        x_log: bool = False,
        y_log: bool = False,
        color=None,
        styles=None,
    ):
        """
        Constructor.
        x_vals can be a list of scalars or a list of tuples/lists. If x_vals is a list
        of scalars and there are multiple x_names, all arguments will have the same value.
        If x_vals is a list of tuples/lists, each element should have the same length as
        x_names.

        :param x_names: Name of the arguments that should appear on the x axis of the plot.
        :type x_names: List[str]
        :param x_vals: List of values to use for the arguments in :code:`x_names`.
        :type x_vals: List[Any]
        :param line_arg: Argument name for which different values correspond to different lines in the plot.
        :type line_arg: str
        :param line_vals: List of values to use for the arguments in :code:`line_arg`.
        :type line_vals: List[Any]
        :param line_names: Label names for the different lines.
        :type line_names: List[str]
        :param plot_name: Name of the plot.
        :type plot_name: str
        :param args: Dictionary of keyword arguments to remain fixed throughout the benchmark.
        :type args: Dict[str, Any]
        :param xlabel: Label for the x axis of the plot.
        :type xlabel: str, optional
        :param ylabel: Label for the y axis of the plot.
        :type ylabel: str, optional
        :param x_log: Whether the x axis should be log scale.
        :type x_log: bool, optional
        :param y_log: Whether the y axis should be log scale.
        :type y_log: bool, optional
        """
        self.x_names = x_names
        self.x_vals = x_vals
        self.x_log = x_log
        self.line_arg = line_arg
        self.line_vals = line_vals
        self.line_names = line_names
        self.y_log = y_log
        self.styles = styles
        # plot info
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_name = plot_name
        self.args = args


class Mark:

    def __init__(self, fn, benchmarks):
        self.fn = fn
        self.benchmarks = benchmarks

    def _run(self, bench: Benchmark, save_path: str, show_plots: bool, print_data: bool, diff_col=False,
             save_precision=6, **kwrags):
        import os

        import matplotlib.pyplot as plt
        import pandas as pd
        y_mean = bench.line_names
        y_min = [f'{x}-min' for x in bench.line_names]
        y_max = [f'{x}-max' for x in bench.line_names]
        x_names = list(bench.x_names)
        df = pd.DataFrame(columns=x_names + y_mean + y_min + y_max)
        for x in bench.x_vals:
            # x can be a single value or a sequence of values.
            if not isinstance(x, (list, tuple)):
                x = [x for _ in x_names]

            if len(x) != len(x_names):
                raise ValueError(f"Expected {len(x_names)} values, got {x}")
            x_args = dict(zip(x_names, x))

            row_mean, row_min, row_max = [], [], []
            for y in bench.line_vals:
                ret = self.fn(**x_args, **{bench.line_arg: y}, **bench.args, **kwrags)
                try:
                    y_mean, y_min, y_max = ret
                except TypeError:
                    y_mean, y_min, y_max = ret, None, None
                row_mean += [y_mean]
                row_min += [y_min]
                row_max += [y_max]
            df.loc[len(df)] = list(x) + row_mean + row_min + row_max

        if bench.plot_name:
            plt.figure()
            ax = plt.subplot()
            # Plot first x value on x axis if there are multiple.
            first_x = x_names[0]
            for i, y in enumerate(bench.line_names):
                y_min, y_max = df[y + '-min'], df[y + '-max']
                col = bench.styles[i][0] if bench.styles else None
                sty = bench.styles[i][1] if bench.styles else None
                ax.plot(df[first_x], df[y], label=y, color=col, ls=sty)
                if not y_min.isnull().all() and not y_max.isnull().all():
                    y_min = y_min.astype(float)
                    y_max = y_max.astype(float)
                    ax.fill_between(df[first_x], y_min, y_max, alpha=0.15, color=col)
            ax.legend()
            ax.set_xlabel(bench.xlabel or first_x)
            ax.set_ylabel(bench.ylabel)
            # ax.set_title(bench.plot_name)
            ax.set_xscale("log" if bench.x_log else "linear")
            ax.set_yscale("log" if bench.y_log else "linear")
            if show_plots:
                plt.show()
            if save_path:
                plt.savefig(os.path.join(save_path, f"{bench.plot_name}.png"))
        df = df[x_names + bench.line_names]
        if diff_col and df.shape[1] == 2:
            col0, col1 = df.columns.tolist()
            df['Diff'] = df[col1] - df[col0]

        if print_data:
            print(bench.plot_name + ':')
            print(df)
        if save_path:
            df.to_csv(os.path.join(save_path, f"{bench.plot_name}.csv"), float_format=f"%.{save_precision}f",
                      index=False)
        return df

    def run(self, show_plots=False, print_data=False, save_path='', return_df=False, **kwargs):
        has_single_bench = isinstance(self.benchmarks, Benchmark)
        benchmarks = [self.benchmarks] if has_single_bench else self.benchmarks
        result_dfs = []
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            html = open(os.path.join(save_path, "results.html"), "w")
            html.write("<html><body>\n")
        for bench in benchmarks:
            result_dfs.append(self._run(bench, save_path, show_plots, print_data, **kwargs))
            if save_path:
                html.write(f"<image src=\"{bench.plot_name}.png\"/>\n")
        if save_path:
            html.write("</body></html>\n")
            html.close()
        if return_df:
            if has_single_bench:
                return result_dfs[0]
            else:
                return result_dfs
        return None


def perf_report(benchmarks):
    """
    Mark a function for benchmarking. The benchmark can then be executed by using the :code:`.run` method on the return value.

    :param benchmarks: Benchmarking configurations.
    :type benchmarks: List of :class:`Benchmark`
    """
    wrapper = lambda fn: Mark(fn, benchmarks)
    return wrapper


def get_dram_gbps(device=None):
    ''' return DRAM bandwidth in GB/s '''
    import torch

    from .runtime import driver
    if not device:
        device = torch.cuda.current_device()
    mem_clock_khz = driver.active.utils.get_device_properties(device)["mem_clock_rate"]  # in kHz
    bus_width = driver.active.utils.get_device_properties(device)["mem_bus_width"]
    bw_gbps = mem_clock_khz * bus_width * 2 / 1e6 / 8  # In GB/s
    return bw_gbps


def get_max_tensorcore_tflops(dtype, clock_rate, device=None):
    import torch

    from .runtime import driver
    if not device:
        device = torch.cuda.current_device()

    num_subcores = driver.active.utils.get_device_properties(device)["multiprocessor_count"] * 4
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8:
        assert dtype == torch.float16
        ops_per_sub_core = 256  # 2 4x4x4 Tensor Cores
    else:
        if dtype in [torch.float32, torch.int32]:
            ops_per_sub_core = 256
        elif dtype in [torch.float16, torch.bfloat16, torch.int16]:
            ops_per_sub_core = 512
        elif dtype in [torch.int8, tl.float8e4nv, tl.float8e4b15, tl.float8e5]:
            ops_per_sub_core = 1024
        else:
            raise RuntimeError("dtype not supported")
    tflops = num_subcores * clock_rate * ops_per_sub_core * 1e-9
    return tflops


# create decorator that wraps test function into
# a cuda-memcheck system call


def cuda_memcheck(**target_kwargs):

    def decorator(test_fn):

        @functools.wraps(test_fn)
        def wrapper(*args, **kwargs):
            import psutil
            ppid_name = psutil.Process(os.getppid()).name()
            run_cuda_memcheck = target_kwargs.items() <= kwargs.items()
            if run_cuda_memcheck and ppid_name != "cuda-memcheck":
                path = os.path.realpath(test_fn.__globals__["__file__"])
                # get path of current file
                env = {"PATH": os.environ["PATH"], "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}
                assert 'request' in kwargs, "memcheck'ed test must have a (possibly unused) `request` fixture"
                test_id = kwargs['request'].node.callspec.id
                cmd = f"{path}::{test_fn.__name__}[{test_id}]"
                out = subprocess.run(["cuda-memcheck", "pytest", "-vs", cmd], capture_output=True, env=env)
                assert out.returncode == 0, "cuda-memcheck returned an error: bounds checking failed"
                assert "ERROR SUMMARY: 0 errors" in str(out.stdout)
            else:
                test_fn(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def set_gpu_clock(ref_sm_clock=1350, ref_mem_clock=1215):
    try:
        subprocess.check_output(["nvidia-smi", "-i", "0", "-pm", "1"])
        subprocess.check_output([
            "nvidia-smi",
            "-i",
            "0",
            f"--lock-gpu-clocks={ref_sm_clock},{ref_sm_clock}",
        ])
        subprocess.check_output([
            "nvidia-smi",
            "-i",
            "0",
            f"--lock-memory-clocks={ref_mem_clock},{ref_mem_clock}",
        ])
        cur_sm_clock = nvsmi(["clocks.current.sm"])[0]
        cur_mem_clock = nvsmi(["clocks.current.memory"])[0]
        assert abs(cur_sm_clock - ref_sm_clock) < 10, f"GPU SMs must run at {ref_sm_clock} MHz"
        assert abs(cur_mem_clock - ref_mem_clock) < 10, f"GPU SMs must run at {ref_mem_clock} MHz"
        tflops = 1e-6 * 2 * 108 * 4 * 256 * ref_sm_clock
        gbps = 640 * 2 * ref_mem_clock * 1e-3
        yield tflops, gbps
    finally:
        subprocess.check_output(["nvidia-smi", "-i", "0", "-pm", "0"])
        subprocess.check_output(["nvidia-smi", "-i", "0", "-rgc"])
        subprocess.check_output(["nvidia-smi", "-i", "0", "-rmc"])


def get_max_simd_tflops(dtype, clock_rate, device=None):
    import torch

    from .runtime import driver
    if not device:
        device = torch.cuda.current_device()

    num_subcores = driver.active.utils.get_device_properties(device)["multiprocessor_count"] * 4
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        if dtype == torch.float32:
            ops_per_sub_core = 32  # 2*16
        elif dtype == torch.float16:
            ops_per_sub_core = 64
        else:
            raise RuntimeError("dtype not supported")
    else:
        if dtype == torch.float32:
            ops_per_sub_core = 32
        elif dtype in [torch.float16, torch.bfloat16]:
            ops_per_sub_core = 64
        else:
            raise RuntimeError("dtype not supported")
    tflops = num_subcores * clock_rate * ops_per_sub_core * 1e-9
    return tflops
