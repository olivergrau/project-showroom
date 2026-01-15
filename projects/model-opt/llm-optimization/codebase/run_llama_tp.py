import os
import json
import argparse
import time
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.distributed as dist

import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, None)
    return default if v is None else int(v)


def init_distributed():
    local_rank = _env_int("LOCAL_RANK", 0)
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1 and not dist.is_initialized():
        deepspeed.init_distributed(dist_backend="nccl")

    return rank, local_rank, world_size


def rank_print(rank: int, msg: str):
    print(f"[rank {rank}] {msg}", flush=True)


def only_rank0_print(rank: int, msg: str):
    if rank == 0:
        print(msg, flush=True)


def _torch_dtype_from_ds_dtype(ds_dtype: str) -> torch.dtype:
    d = (ds_dtype or "").lower()
    if d in ("fp16", "float16", "half"):
        return torch.float16
    if d in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype in DeepSpeed config: {ds_dtype!r}")


def load_model_and_tokenizer(model_id: str, torch_dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    model.eval()
    return model, tokenizer


def read_and_fix_ds_config(
    ds_config_path: str,
    world_size: int,
    rank: int,
    local_rank: int,
    force_bf16: bool,
):
    with open(ds_config_path, "r") as f:
        ds_cfg = json.load(f) or {}

    if "tensor_parallel" not in ds_cfg or ds_cfg["tensor_parallel"] is None:
        ds_cfg["tensor_parallel"] = {}

    cfg_tp_size = int(ds_cfg["tensor_parallel"].get("tp_size", world_size))
    if cfg_tp_size != world_size:
        if rank == 0:
            print(
                f"[rank 0] WARNING: ds config tp_size={cfg_tp_size} but WORLD_SIZE={world_size}. "
                f"Overriding tp_size -> {world_size} for this run.",
                flush=True,
            )
        ds_cfg["tensor_parallel"]["tp_size"] = world_size

    ds_dtype = (ds_cfg.get("dtype") or "fp16").lower()

    if force_bf16:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(local_rank)
            if major >= 8:
                ds_dtype = "bf16"
                ds_cfg["dtype"] = "bf16"
            else:
                if rank == 0:
                    print(
                        f"[rank 0] WARNING: --use_bf16 requested but device capability is {major}.{minor}. "
                        f"Keeping fp16.",
                        flush=True,
                    )
                ds_dtype = "fp16"
                ds_cfg["dtype"] = "fp16"
        else:
            ds_dtype = "fp16"
            ds_cfg["dtype"] = "fp16"

    torch_dtype = _torch_dtype_from_ds_dtype(ds_dtype)
    return ds_cfg, ds_dtype, torch_dtype


def apply_deepspeed_tp(model, ds_cfg: dict):
    # IMPORTANT: do not pass dtype kwarg if dtype is in config
    engine = deepspeed.init_inference(model, config=ds_cfg)
    return engine


def run_generate(engine, tokenizer, prompt: str, max_new_tokens: int) -> Tuple[torch.Tensor, int, int]:
    """
    Runs deterministic greedy generation for ONE prompt.
    Returns:
      out_ids: (1, seq_len_total)
      input_len: prompt token count
      gen_tokens: number of new tokens appended
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(torch.cuda.current_device()) for k, v in inputs.items()}

    input_len = int(inputs["input_ids"].shape[1])

    with torch.no_grad():
        out_ids = engine.generate(
            **inputs,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    out_len = int(out_ids.shape[1])
    gen_tokens = max(0, out_len - input_len)
    return out_ids, input_len, gen_tokens


def _broadcast_object(obj, rank: int, world_size: int):
    """
    Broadcast a picklable python object from rank 0 to all ranks.
    """
    if world_size == 1:
        return obj
    obj_list = [obj] if rank == 0 else [None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


def load_prompts_for_benchmark(
    benchmark_json: str,
    benchmark_n: int,
    rank: int,
    world_size: int,
) -> List[str]:
    prompts = None
    if rank == 0:
        with open(benchmark_json, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        if not isinstance(prompts, list) or not all(isinstance(x, str) for x in prompts):
            raise ValueError(f"{benchmark_json} must be a JSON list of strings")

        if benchmark_n > 0:
            prompts = prompts[:benchmark_n]

    prompts = _broadcast_object(prompts, rank=rank, world_size=world_size)

    if prompts is None:
        raise RuntimeError("Failed to broadcast prompts")
    return prompts


def benchmark_generate(
    engine,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    warmup: int,
    rank: int,
    world_size: int,
    dump_n: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Benchmarks engine.generate() over prompts.

    Key change vs previous version:
    - While benchmarking, rank 0 can CAPTURE the generated token IDs for a small subset
      (first dump_n prompts) without decoding inside the timed loop.
    - The captured outputs are returned so rank 0 can decode and write them AFTER timing.

    Returns:
      metrics: dict
      captured: list (rank-0 only, otherwise empty)
    """
    # Warmup: all ranks must participate
    for _ in range(max(0, warmup)):
        _ = run_generate(engine, tokenizer, prompts[0], max_new_tokens=max_new_tokens)
        if world_size > 1:
            dist.barrier()

    # Timed loop: all ranks participate; rank 0 measures wall time
    if world_size > 1:
        dist.barrier()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if rank == 0:
        t0 = time.time()

    total_gen_tokens_rank0 = 0
    captured: List[Dict[str, Any]] = []

    for i, p in enumerate(prompts):
        out_ids, input_len, gen_tokens = run_generate(engine, tokenizer, p, max_new_tokens=max_new_tokens)

        if rank == 0:
            total_gen_tokens_rank0 += int(gen_tokens)

            # Capture only a few samples, and do NOT decode here.
            if dump_n and i < dump_n:
                # store 1D tensor with full sequence (prompt + generated)
                captured.append(
                    {
                        "i": int(i),
                        "prompt": p,
                        "input_len": int(input_len),
                        "out_ids_tensor": out_ids[0].detach(),
                    }
                )

        # Keep ranks in lockstep for TP correctness and stable timing
        if world_size > 1:
            dist.barrier()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if rank == 0:
        total_time = time.time() - t0
    else:
        total_time = None

    total_time = _broadcast_object(total_time, rank=rank, world_size=world_size)

    n = len(prompts)
    mean_latency = float(total_time) / max(1, n)
    throughput_tok_s = float(total_gen_tokens_rank0) / max(1e-9, float(total_time))

    metrics = {
        "n_prompts": n,
        "total_time_s": float(total_time),
        "mean_latency_s": float(mean_latency),
        "total_gen_tokens_rank0": int(total_gen_tokens_rank0),
        "throughput_tokens_per_s_rank0": float(throughput_tok_s),
    }
    return metrics, captured


def _finalize_and_write_dump_tp(
    out_path: str,
    tokenizer,
    result_payload: Dict[str, Any],
    captured: List[Dict[str, Any]],
    dump_max_new_tokens: int,
):
    """
    Convert captured tensors to token-id lists and decode text on rank 0,
    then write ONE JSON containing benchmark metrics + sample outputs.

    IMPORTANT:
    - Decoding happens outside the timed section, so the benchmark remains clean.
    """
    items: List[Dict[str, Any]] = []
    gen_lens: List[int] = []

    for c in captured:
        out_ids_1d = c["out_ids_tensor"]
        if isinstance(out_ids_1d, torch.Tensor) and out_ids_1d.is_cuda:
            out_ids_1d = out_ids_1d.cpu()
        out_ids_list = out_ids_1d.tolist() if isinstance(out_ids_1d, torch.Tensor) else list(out_ids_1d)

        input_len = int(c["input_len"])
        gen_ids = out_ids_list[input_len : input_len + int(dump_max_new_tokens)]
        gen_lens.append(len(gen_ids))

        # Full decoded text (prompt + completion), helps sanity checking
        gen_text_full = tokenizer.decode(out_ids_list, skip_special_tokens=True)

        items.append(
            {
                "i": int(c["i"]),
                "prompt": c["prompt"],
                "input_len": input_len,
                "gen_token_ids": gen_ids,
                "gen_text": gen_text_full,
            }
        )

    dump_stats = {
        "n_dumped": int(len(items)),
        "dump_max_new_tokens": int(dump_max_new_tokens),
        "gen_tokens_min": int(min(gen_lens)) if gen_lens else 0,
        "gen_tokens_max": int(max(gen_lens)) if gen_lens else 0,
        "gen_tokens_mean": float(sum(gen_lens) / max(1, len(gen_lens))) if gen_lens else 0.0,
    }

    payload = {
        **result_payload,
        "dump": {
            "stats": dump_stats,
            "items": items,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed launcher injected arg")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--deepspeed_config", type=str, default="ds_tp.json")

    parser.add_argument("--prompt", type=str, default="Write one short news headline about a rocket launch.")
    parser.add_argument("--max_new_tokens", type=int, default=16)

    parser.add_argument("--benchmark_json", type=str, default=None, help="Path to JSON list of prompts")
    parser.add_argument("--benchmark_n", type=int, default=25, help="How many prompts to benchmark from JSON")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations before timing")

    parser.add_argument("--dump_outputs_json", type=str, default=None)
    parser.add_argument("--dump_n", type=int, default=5)
    parser.add_argument("--dump_max_new_tokens", type=int, default=None)

    parser.add_argument("--use_bf16", action="store_true", help="Force bf16 if supported by the GPU.")
    args = parser.parse_args()

    rank, local_rank, world_size = init_distributed()

    rank_print(rank, f"dist.is_initialized={dist.is_initialized()}")
    if dist.is_initialized():
        rank_print(rank, f"dist.get_world_size()={dist.get_world_size()}")

    device_str = "cpu"
    if torch.cuda.is_available():
        device_str = f"cuda:{local_rank} name={torch.cuda.get_device_name(local_rank)}"
    rank_print(rank, f"world_size={world_size}, local_rank={local_rank}, device={device_str}")

    ds_cfg, ds_dtype, torch_dtype = read_and_fix_ds_config(
        args.deepspeed_config,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        force_bf16=args.use_bf16,
    )

    rank_print(rank, f"DeepSpeed dtype={ds_dtype}, torch_dtype(load)={torch_dtype}")
    rank_print(rank, f"DeepSpeed tensor_parallel config: {ds_cfg.get('tensor_parallel', {})}")

    model, tokenizer = load_model_and_tokenizer(args.model_id, torch_dtype=torch_dtype)
    engine = apply_deepspeed_tp(model, ds_cfg)

    # Benchmark mode
    if args.benchmark_json:
        prompts = load_prompts_for_benchmark(
            args.benchmark_json,
            benchmark_n=args.benchmark_n,
            rank=rank,
            world_size=world_size,
        )

        # Only rank 0 captures samples, but all ranks run the same generate loop.
        dump_n = 0
        dump_tokens = args.max_new_tokens
        if args.dump_outputs_json and rank == 0:
            dump_n = max(1, min(int(args.dump_n), len(prompts)))
            dump_tokens = int(args.dump_max_new_tokens or args.max_new_tokens)

        metrics, captured = benchmark_generate(
            engine,
            tokenizer,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            warmup=args.warmup,
            rank=rank,
            world_size=world_size,
            dump_n=dump_n,
        )

        if rank == 0:
            # Print as JSON for easy copy/paste into notebook tables
            result = {
                "mode": "tp",
                "world_size": world_size,
                "tp_size": int(ds_cfg.get("tensor_parallel", {}).get("tp_size", world_size)),
                "dtype": ds_dtype,
                **metrics,
            }
            print("\n=== TP BENCHMARK RESULT (rank 0, JSON) ===")
            print(json.dumps(result, indent=2))
            print()

            if args.dump_outputs_json:
                print("Writing benchmark outputs (captured during timed run).")
                _finalize_and_write_dump_tp(
                    out_path=args.dump_outputs_json,
                    tokenizer=tokenizer,
                    result_payload=result,
                    captured=captured,
                    dump_max_new_tokens=dump_tokens,
                )
                print(f"Wrote outputs to: {args.dump_outputs_json}", flush=True)

    else:
        # Simple smoke run (still all ranks participate, but output printed only by rank 0)
        if world_size > 1:
            dist.barrier()

        out_ids, _, _ = run_generate(engine, tokenizer, args.prompt, args.max_new_tokens)

        if rank == 0:
            text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            only_rank0_print(rank, "\n=== TP RESULT (rank 0) ===")
            only_rank0_print(rank, f"Output:\n{text}\n")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
