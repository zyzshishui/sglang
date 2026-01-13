#!/usr/bin/env python3
"""
Profile multi-LoRA diffusion requests with optional torch profiler.

Example:
  python scripts/profile_multilora_diffusion.py \\
    --model Tongyi-MAI/Z-Image-Turbo \\
    --lora1 reverentelusarca/elusarca-anime-style-lora-z-image-turbo \\
    --lora2 tarn59/pixel_art_style_lora_z_image_turbo \\
    --prompt "Doraemon is eating dorayaki" \\
    --size 1024x1024 \\
    --output outputs/AMD.json \\
    --warmup \\
    --warmup-resolutions 1024x1024 \\
    --profile-target multi
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

from sglang.multimodal_gen.test.server.test_server_utils import ServerManager
from sglang.multimodal_gen.test.server.testcase_configs import (
    BASELINE_CONFIG,
    PerformanceSummary,
)
from sglang.multimodal_gen.test.test_utils import (
    get_dynamic_server_port,
    wait_for_req_perf_record,
)


def _git_rev(path: str) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).decode().strip()
    except Exception:
        return "unknown"


def _post_image(
    base_url: str,
    model: str,
    prompt: str,
    size: str,
    profile: bool,
    num_profiled_timesteps: int,
    profile_all_stages: bool,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "n": 1,
        "response_format": "b64_json",
    }
    if profile:
        payload.update(
            {
                "profile": True,
                "num_profiled_timesteps": num_profiled_timesteps,
                "profile_all_stages": profile_all_stages,
            }
        )
    resp = requests.post(f"{base_url}/images/generations", json=payload, timeout=1800)
    resp.raise_for_status()
    return resp.json()["id"]


def _generate_and_collect(
    base_url: str,
    perf_log_path: Path,
    label: str,
    model: str,
    prompt: str,
    size: str,
    profile: bool,
    num_profiled_timesteps: int,
    profile_all_stages: bool,
) -> dict:
    request_id = _post_image(
        base_url,
        model=model,
        prompt=prompt,
        size=size,
        profile=profile,
        num_profiled_timesteps=num_profiled_timesteps,
        profile_all_stages=profile_all_stages,
    )
    record = wait_for_req_perf_record(request_id, perf_log_path, timeout=1800)
    if record is None:
        raise RuntimeError(f"No perf record for {label} request_id={request_id}")

    summary = PerformanceSummary.from_req_perf_record(record, BASELINE_CONFIG.step_fractions)
    return {
        "request_id": request_id,
        "e2e_ms": round(summary.e2e_ms, 2),
        "avg_denoise_ms": round(summary.avg_denoise_ms, 2),
        "median_denoise_ms": round(summary.median_denoise_ms, 2),
        "stage_metrics_ms": {k: round(v, 2) for k, v in summary.stage_metrics.items()},
    }


def _warmup_requests(
    base_url: str,
    count: int,
    model: str,
    prompt: str,
    size: str,
) -> None:
    for _ in range(count):
        _post_image(
            base_url,
            model=model,
            prompt=prompt,
            size=size,
            profile=False,
            num_profiled_timesteps=0,
            profile_all_stages=False,
        )


def _set_lora(base_url: str, payload: dict) -> float:
    start = time.perf_counter()
    resp = requests.post(f"{base_url}/set_lora", json=payload, timeout=1800)
    if resp.status_code != 200:
        raise RuntimeError(f"set_lora failed: {resp.status_code} {resp.text}")
    return (time.perf_counter() - start) * 1000.0


def _profile_enabled(target: str, name: str) -> bool:
    if target == "none":
        return False
    if target == "all":
        return True
    return target == name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile multi-LoRA diffusion requests with optional torch profiler."
    )
    parser.add_argument("--model", default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument(
        "--lora1",
        default="reverentelusarca/elusarca-anime-style-lora-z-image-turbo",
    )
    parser.add_argument("--lora2", default="tarn59/pixel_art_style_lora_z_image_turbo")
    parser.add_argument("--prompt", default="Doraemon is eating dorayaki")
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--output", default="outputs/multi_lora_profile.json")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--warmup-resolutions", default=None)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--lora-warmup-requests", type=int, default=1)
    parser.add_argument(
        "--profile-target",
        choices=["none", "baseline", "single", "multi", "all"],
        default="none",
        help="Which request(s) to profile.",
    )
    parser.add_argument("--profile-all-stages", action="store_true")
    parser.add_argument("--num-profiled-timesteps", type=int, default=5)
    parser.add_argument("--server-extra-args", default="")
    parser.add_argument("--pythonpath", default=os.environ.get("PYTHONPATH", ""))
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser.add_argument("--hip-visible-devices", default=os.environ.get("HIP_VISIBLE_DEVICES"))
    parser.add_argument("--rocr-visible-devices", default=os.environ.get("ROCR_VISIBLE_DEVICES"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    port = args.port or get_dynamic_server_port()

    extra_args = f"--num-gpus {args.num_gpus}"
    if args.warmup:
        extra_args += " --warmup"
        if args.warmup_resolutions:
            extra_args += f" --warmup-resolutions {args.warmup_resolutions}"
    if args.server_extra_args:
        extra_args += f" {args.server_extra_args}"

    env_vars: dict[str, str] = {}
    if args.pythonpath:
        env_vars["PYTHONPATH"] = args.pythonpath
    if args.cuda_visible_devices is not None:
        env_vars["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.hip_visible_devices is not None:
        env_vars["HIP_VISIBLE_DEVICES"] = args.hip_visible_devices
    if args.rocr_visible_devices is not None:
        env_vars["ROCR_VISIBLE_DEVICES"] = args.rocr_visible_devices

    mgr = ServerManager(
        model=args.model,
        port=port,
        extra_args=extra_args,
        env_vars=env_vars,
    )

    results: dict[str, object] = {
        "model": args.model,
        "lora_paths": [args.lora1, args.lora2],
        "prompt": args.prompt,
        "size": args.size,
        "port": port,
        "commit": _git_rev(str(Path(__file__).resolve().parents[1])),
    }

    ctx: Optional[object] = None
    try:
        ctx = mgr.start()
        base_url = f"http://localhost:{ctx.port}/v1"

        if args.warmup_requests > 0:
            _warmup_requests(base_url, args.warmup_requests, args.model, args.prompt, args.size)

        results["baseline"] = _generate_and_collect(
            base_url,
            ctx.perf_log_path,
            "baseline",
            args.model,
            args.prompt,
            args.size,
            _profile_enabled(args.profile_target, "baseline"),
            args.num_profiled_timesteps,
            args.profile_all_stages,
        )

        results["set_lora_single_ms"] = round(
            _set_lora(base_url, {"lora_nickname": "lora1", "lora_path": args.lora1}), 2
        )
        if args.lora_warmup_requests > 0:
            _warmup_requests(
                base_url,
                args.lora_warmup_requests,
                args.model,
                args.prompt,
                args.size,
            )
        results["single_lora"] = _generate_and_collect(
            base_url,
            ctx.perf_log_path,
            "single_lora",
            args.model,
            args.prompt,
            args.size,
            _profile_enabled(args.profile_target, "single"),
            args.num_profiled_timesteps,
            args.profile_all_stages,
        )

        results["set_lora_second_ms"] = round(
            _set_lora(base_url, {"lora_nickname": "lora2", "lora_path": args.lora2}), 2
        )

        results["set_lora_multi_ms"] = round(
            _set_lora(
                base_url,
                {
                    "lora_nickname": ["lora1", "lora2"],
                    "lora_path": [args.lora1, args.lora2],
                    "target": "all",
                    "strength": [1.0, 1.0],
                },
            ),
            2,
        )
        if args.lora_warmup_requests > 0:
            _warmup_requests(
                base_url,
                args.lora_warmup_requests,
                args.model,
                args.prompt,
                args.size,
            )
        results["multi_lora"] = _generate_and_collect(
            base_url,
            ctx.perf_log_path,
            "multi_lora",
            args.model,
            args.prompt,
            args.size,
            _profile_enabled(args.profile_target, "multi"),
            args.num_profiled_timesteps,
            args.profile_all_stages,
        )

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
        print("Torch profiler traces are saved under ./logs/ on the server host.")
    finally:
        if ctx is not None:
            ctx.cleanup()


if __name__ == "__main__":
    main()
