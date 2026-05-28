#!/usr/bin/env python3
"""Bootstrap an isolated repo-faithful RadGPT vLLM server environment.

This script exists so the main training environment never needs to be mutated
just to run the RadGPT evaluation backend.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_MAGISTERKA_ROOT = Path("/net/scratch/hscra/plgrid/plgikolton/Magisterka")
DEFAULT_MAIN_VENV = DEFAULT_MAGISTERKA_ROOT / ".venv"
DEFAULT_SERVER_VENV = DEFAULT_MAGISTERKA_ROOT / ".venv-radgpt-vllm"
DEFAULT_VLLM_ROOT = DEFAULT_MAGISTERKA_ROOT / "vllm-gh200"
DEFAULT_RADGPT_ROOT = DEFAULT_MAGISTERKA_ROOT / "RadGPT"
DEFAULT_OUTPUT_ROOT = DEFAULT_MAGISTERKA_ROOT / "Masters" / "outputs" / "decoder" / "benchmark_val_10pct_best_vs_last"
DEFAULT_VLLM_TAG = "v0.6.1.post2"
DEFAULT_VLLM_MODEL = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
DEFAULT_RADGPT_TRANSFORMERS_REF = "git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830"


CUSTOM_CACHE_MANAGER_SOURCE = """import os

from triton.runtime.cache import FileCacheManager
import triton.runtime.cache as _triton_cache

from vllm.logger import init_logger

logger = init_logger(__name__)


def _default_cache_dir():
    fn = getattr(_triton_cache, "default_cache_dir", None)
    if callable(fn):
        return fn()
    return _triton_cache.knobs.cache.dir


def _default_dump_dir():
    fn = getattr(_triton_cache, "default_dump_dir", None)
    if callable(fn):
        return fn()
    return _triton_cache.knobs.cache.dump_dir


def _default_override_dir():
    fn = getattr(_triton_cache, "default_override_dir", None)
    if callable(fn):
        return fn()
    return _triton_cache.knobs.cache.override_dir


def maybe_set_triton_cache_manager() -> None:
    cache_manger = os.environ.get("TRITON_CACHE_MANAGER", None)
    if cache_manger is None:
        manager = "vllm.triton_utils.custom_cache_manager:CustomCacheManager"
        logger.info("Setting Triton cache manager to: %s", manager)
        os.environ["TRITON_CACHE_MANAGER"] = manager


class CustomCacheManager(FileCacheManager):
    def __init__(self, key, override=False, dump=False):
        self.key = key
        self.lock_path = None
        if dump:
            self.cache_dir = _default_dump_dir()
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)
        elif override:
            self.cache_dir = _default_override_dir()
            self.cache_dir = os.path.join(self.cache_dir, self.key)
        else:
            self.cache_dir = os.getenv("TRITON_CACHE_DIR", "").strip() or _default_cache_dir()
            if self.cache_dir:
                self.cache_dir = f"{self.cache_dir}_{os.getpid()}"
                self.cache_dir = os.path.join(self.cache_dir, self.key)
                self.lock_path = os.path.join(self.cache_dir, "lock")
                os.makedirs(self.cache_dir, exist_ok=True)
            else:
                raise RuntimeError("Could not create or locate cache dir")
"""

TORCH_SDPA_SELECTOR_PATCH_OLD = """    elif backend == _Backend.TORCH_SDPA:
        assert is_cpu(), RuntimeError(
            "Torch SDPA backend is only used for the CPU device.")
        logger.info("Using Torch SDPA backend.")
        from vllm.attention.backends.torch_sdpa import TorchSDPABackend
        return TorchSDPABackend
"""

TORCH_SDPA_SELECTOR_PATCH_NEW = """    elif backend == _Backend.TORCH_SDPA:
        logger.info("Using Torch SDPA backend.")
        from vllm.attention.backends.torch_sdpa import TorchSDPABackend
        return TorchSDPABackend
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-venv", default=str(DEFAULT_MAIN_VENV))
    parser.add_argument("--server-venv", default=str(DEFAULT_SERVER_VENV))
    parser.add_argument("--vllm-root", default=str(DEFAULT_VLLM_ROOT))
    parser.add_argument("--radgpt-root", default=str(DEFAULT_RADGPT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--vllm-tag", default=DEFAULT_VLLM_TAG)
    parser.add_argument("--model", default=DEFAULT_VLLM_MODEL)
    parser.add_argument("--radgpt-transformers-ref", default=DEFAULT_RADGPT_TRANSFORMERS_REF)
    parser.add_argument("--cuda-home", default="/net/software/aarch64/el9/CUDA/12.4.0")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=120000)
    parser.add_argument("--dtype", default="half")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cuda-visible-devices", default="0,1")
    parser.add_argument("--max-jobs", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--skip-clone", action="store_true", help="Do not recreate the copied server venv from the main venv.")
    parser.add_argument("--skip-install", action="store_true", help="Do not reinstall vLLM into the copied env.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip the post-install import verification.")
    parser.add_argument("--launch", action="store_true", help="Launch the repo-faithful RadGPT vLLM API after setup.")
    args = parser.parse_args()

    main_venv = Path(args.main_venv).expanduser().resolve()
    server_venv = Path(args.server_venv).expanduser().resolve()
    vllm_root = Path(args.vllm_root).expanduser().resolve()
    radgpt_root = Path(args.radgpt_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    api_log = output_root / "radgpt_api.log"
    hf_cache = radgpt_root / "HFCache"
    hf_cache.mkdir(parents=True, exist_ok=True)

    if not args.skip_clone:
        recreate_copied_env(main_venv=main_venv, server_venv=server_venv)

    ensure_vllm_repo(vllm_root=vllm_root, tag=args.vllm_tag)

    if not args.skip_install:
        install_server_stack(
            server_venv=server_venv,
            vllm_root=vllm_root,
            cuda_home=args.cuda_home,
            max_jobs=args.max_jobs,
            radgpt_transformers_ref=str(args.radgpt_transformers_ref),
        )
        patch_triton_cache(server_venv=server_venv)
        patch_attention_selector(server_venv=server_venv)

    if not args.skip_verify:
        verify_server_env(server_venv=server_venv, cuda_home=args.cuda_home)

    print("\nSetup finished.")
    print(f"Copied server env: {server_venv}")
    print(f"vLLM repo: {vllm_root}")
    print(f"API log: {api_log}")

    if args.launch:
        launch_server(
            server_venv=server_venv,
            radgpt_root=radgpt_root,
            model=args.model,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            port=args.port,
            cuda_visible_devices=args.cuda_visible_devices,
            cuda_home=args.cuda_home,
            log_path=api_log,
            timeout_seconds=args.timeout_seconds,
        )


def recreate_copied_env(*, main_venv: Path, server_venv: Path) -> None:
    if not main_venv.is_dir():
        raise FileNotFoundError(f"Main venv not found: {main_venv}")
    if server_venv.exists():
        shutil.rmtree(server_venv)
    run(["rsync", "-a", f"{main_venv}/", f"{server_venv}/"])


def ensure_vllm_repo(*, vllm_root: Path, tag: str) -> None:
    if not (vllm_root / ".git").is_dir():
        run(["git", "clone", "--branch", tag, "--depth", "1", "https://github.com/vllm-project/vllm.git", str(vllm_root)])
    run(["git", "fetch", "--tags"], cwd=vllm_root)
    run(["git", "checkout", tag], cwd=vllm_root)


def install_server_stack(
    *,
    server_venv: Path,
    vllm_root: Path,
    cuda_home: str,
    max_jobs: int,
    radgpt_transformers_ref: str,
) -> None:
    python_bin = server_venv / "bin" / "python"
    env = build_env(server_venv=server_venv, cuda_home=cuda_home, max_jobs=max_jobs)
    run([str(python_bin), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], env=env)
    run([str(python_bin), "-m", "pip", "install", "cmake", "ninja", "packaging", "jinja2", "numpy<2.0.0"], env=env)
    run([str(python_bin), "-m", "pip", "install", "-r", "requirements-common.txt"], cwd=vllm_root, env=env)
    run([str(python_bin), "-m", "pip", "install", "ray>=2.9", "nvidia-ml-py"], env=env)
    build_log = vllm_root / "build.log"
    with build_log.open("w", encoding="utf-8") as handle:
        run(
            [str(python_bin), "-m", "pip", "install", "--no-build-isolation", "--no-deps", "."],
            cwd=vllm_root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    # Match the upstream RadGPT README's transformers stack rather than inheriting
    # whatever newer version happens to live in the main training environment.
    run([str(python_bin), "-m", "pip", "install", str(radgpt_transformers_ref)], env=env)


def patch_triton_cache(*, server_venv: Path) -> None:
    cache_path = server_venv / "lib" / "python3.11" / "site-packages" / "vllm" / "triton_utils" / "custom_cache_manager.py"
    cache_path.write_text(CUSTOM_CACHE_MANAGER_SOURCE, encoding="utf-8")


def patch_attention_selector(*, server_venv: Path) -> None:
    selector_path = server_venv / "lib" / "python3.11" / "site-packages" / "vllm" / "attention" / "selector.py"
    text = selector_path.read_text(encoding="utf-8")
    if TORCH_SDPA_SELECTOR_PATCH_OLD in text:
        text = text.replace(TORCH_SDPA_SELECTOR_PATCH_OLD, TORCH_SDPA_SELECTOR_PATCH_NEW, 1)
    selector_path.write_text(text, encoding="utf-8")


def verify_server_env(*, server_venv: Path, cuda_home: str) -> None:
    python_bin = server_venv / "bin" / "python"
    env = build_env(server_venv=server_venv, cuda_home=cuda_home, max_jobs=8)
    code = textwrap.dedent(
        """
        import numpy
        import triton
        import vllm
        import torch
        print("numpy", numpy.__version__)
        print("triton", triton.__version__)
        print("vllm", vllm.__version__)
        print("torch", torch.__version__, "cuda=", torch.version.cuda)
        """
    )
    run([str(python_bin), "-c", code], env=env)


def launch_server(
    *,
    server_venv: Path,
    radgpt_root: Path,
    model: str,
    dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    port: int,
    cuda_visible_devices: str,
    cuda_home: str,
    log_path: Path,
    timeout_seconds: int,
) -> None:
    python_bin = server_venv / "bin" / "python"
    env = build_env(server_venv=server_venv, cuda_home=cuda_home, max_jobs=8)
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    env["TRANSFORMERS_CACHE"] = str(radgpt_root / "HFCache")
    env["HF_HOME"] = str(radgpt_root / "HFCache")
    env["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"
    cmd = [
        str(python_bin),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model),
        "--dtype",
        str(dtype),
        "--tensor-parallel-size",
        str(int(tensor_parallel_size)),
        "--gpu-memory-utilization",
        str(float(gpu_memory_utilization)),
        "--port",
        str(int(port)),
        "--max-model-len",
        str(int(max_model_len)),
        "--enable-chunked-prefill=False",
        "--enforce-eager",
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=str(radgpt_root),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    wait_for_server(port=port, timeout_seconds=timeout_seconds, process=process, log_path=log_path)
    print(f"\nRadGPT API is ready at http://127.0.0.1:{port}/v1")
    print(f"PID: {process.pid}")


def wait_for_server(*, port: int, timeout_seconds: int, process: subprocess.Popen[str], log_path: Path) -> None:
    deadline = time.time() + max(1, int(timeout_seconds))
    url = f"http://127.0.0.1:{int(port)}/v1/models"
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "vLLM API process exited before becoming ready.\n\n"
                + tail_text(log_path)
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            pass
        time.sleep(5)
    raise RuntimeError(
        "Timed out waiting for the vLLM API.\n\n"
        + tail_text(log_path)
    )


def build_env(*, server_venv: Path, cuda_home: str, max_jobs: int) -> dict[str, str]:
    env = os.environ.copy()
    torch_lib = server_venv / "lib" / "python3.11" / "site-packages" / "torch" / "lib"
    env["CUDA_HOME"] = str(cuda_home)
    env["PATH"] = f"{cuda_home}/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{torch_lib}:{cuda_home}/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    env["NCCL_P2P_DISABLE"] = "1"
    env["MAX_JOBS"] = str(int(max_jobs))
    env["VLLM_TARGET_DEVICE"] = "cuda"
    env.pop("PIP_EXTRA_INDEX_URL", None)
    env["PIP_INDEX_URL"] = "https://pypi.org/simple"
    return env


def tail_text(path: Path, *, line_count: int = 80) -> str:
    if not path.is_file():
        return f"(missing log file: {path})"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdout: object | None = None,
    stderr: object | None = None,
) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"\n$ {printable}")
    subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=stdout,
        stderr=stderr,
        check=True,
    )


if __name__ == "__main__":
    main()
