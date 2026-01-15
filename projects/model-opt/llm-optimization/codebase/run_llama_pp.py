import os
import json
import argparse
import time
import inspect
from typing import List, Optional, Tuple, Any, Dict

import torch
import torch.distributed as dist

import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM

from typing import Optional, Tuple, Any, List


def make_causal_mask_4d(q_len: int, k_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Make sure dtype matches the model's q/k dtype for SDPA
    m = torch.zeros((q_len, k_len), device=device, dtype=dtype)
    if k_len > 0:
        future = torch.triu(torch.ones((q_len, k_len), device=device, dtype=torch.bool), diagonal=1)
        m = m.masked_fill(future, torch.finfo(dtype).min)  # safer than -inf in fp16
    return m.view(1, 1, q_len, k_len)

import inspect

@torch.no_grad()
def stage_forward_nocache(
    model,
    stage_id: int,
    pp_size: int,
    input_ids,
    hidden_in,
    position_ids,
    attention_mask_4d,
):
    num_layers = len(model.model.layers)
    start, end = compute_layer_partition(num_layers, pp_size, stage_id)
    layers = model.model.layers[start:end]

    if stage_id == 0:
        assert input_ids is not None
        hidden_states = model.model.embed_tokens(input_ids)
    else:
        assert hidden_in is not None
        hidden_states = hidden_in

    # Prepare position_embeddings ONLY if this Transformers version/layer needs it.
    # In your stack trace, LlamaAttention unpacks (cos, sin) from position_embeddings,
    # so this must be provided.
    position_embeddings = None
    needs_pos_emb = False
    try:
        sig0 = inspect.signature(layers[0].forward)
        needs_pos_emb = "position_embeddings" in sig0.parameters
    except Exception:
        needs_pos_emb = False

    if needs_pos_emb:
        rotary_emb = getattr(model.model, "rotary_emb", None)
        if rotary_emb is None:
            rotary_emb = layers[0].self_attn.rotary_emb

        # IMPORTANT: compute once per call. Must return (cos, sin) tuple.
        position_embeddings = rotary_emb(hidden_states, position_ids)

    for layer in layers:
        kwargs = dict(
            attention_mask=attention_mask_4d,
            position_ids=position_ids,
            use_cache=False,
        )
        # Only pass position_embeddings if the layer accepts it.
        try:
            sig = inspect.signature(layer.forward)
            if "position_embeddings" in sig.parameters:
                kwargs["position_embeddings"] = position_embeddings
        except Exception:
            # If introspection fails, don't pass it.
            pass

        out = layer(hidden_states, **kwargs)
        hidden_states = out[0] if isinstance(out, (tuple, list)) else out

    return hidden_states


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, None)
    return default if v is None else int(v)


def init_distributed() -> Tuple[int, int, int]:
    local_rank = _env_int("LOCAL_RANK", 0)
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Only initialize process group when truly distributed
    if world_size > 1 and not dist.is_initialized():
        deepspeed.init_distributed(dist_backend="nccl")

    return rank, local_rank, world_size


def rank_print(rank: int, msg: str):
    print(f"[rank {rank}] {msg}", flush=True)


def barrier_if_dist():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _torch_dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {s!r}")


def _broadcast_object(obj, rank: int, world_size: int):
    if world_size == 1:
        return obj
    buf = [obj] if rank == 0 else [None]
    dist.broadcast_object_list(buf, src=0)
    return buf[0]


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
    prompts = _broadcast_object(prompts, rank, world_size)
    if prompts is None:
        raise RuntimeError("Failed to broadcast prompts")
    return prompts


def compute_layer_partition(num_layers: int, pp_size: int, stage_id: int) -> Tuple[int, int]:
    base = num_layers // pp_size
    rem = num_layers % pp_size
    start = stage_id * base + min(stage_id, rem)
    end = start + base + (1 if stage_id < rem else 0)
    return start, end


def read_pp_config(ds_pp_path: str, world_size: int) -> Tuple[str, int]:
    with open(ds_pp_path, "r") as f:
        cfg = json.load(f) or {}
    dtype = (cfg.get("dtype") or "fp16").lower()

    # Always force pp_size to world_size for correctness
    pp_size = world_size
    return dtype, pp_size

@torch.inference_mode()
def last_stage_logits(model, hidden: torch.Tensor) -> torch.Tensor:
    core = model.model
    hidden = core.norm(hidden)
    logits = model.lm_head(hidden)
    return logits


def send_tensor(t: torch.Tensor, dst: int):
    dist.send(t.contiguous(), dst=dst)


def recv_tensor(shape: Tuple[int, ...], dtype: torch.dtype, src: int, device: torch.device) -> torch.Tensor:
    t = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(t, src=src)
    return t


@torch.inference_mode()
def greedy_decode_full_model(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> int:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)

    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    last_tok = input_ids[:, -1:]  # (1,1)

    for _ in range(max_new_tokens):
        out2 = model(input_ids=last_tok, past_key_values=past, use_cache=True)
        past = out2.past_key_values
        logits = out2.logits[:, -1, :]
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)  # (1,1)
        last_tok = next_tok

    return max_new_tokens


@torch.inference_mode()
def greedy_decode_full_model_capture(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> Tuple[List[int], List[int]]:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)  # (1,S)
    prompt_ids = input_ids[0].tolist()

    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    last_tok = input_ids[:, -1:]  # (1,1)

    gen_ids: List[int] = []
    for _ in range(max_new_tokens):
        out2 = model(input_ids=last_tok, past_key_values=past, use_cache=True)
        past = out2.past_key_values
        logits = out2.logits[:, -1, :]
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)  # (1,1)
        gen_ids.append(int(next_tok.item()))
        last_tok = next_tok

    return prompt_ids, gen_ids


def _finalize_and_write_dump_pp(
    out_path: str,
    tokenizer,
    result_payload: Dict[str, Any],
    captured: List[Dict[str, Any]],
    dump_max_new_tokens: int,
):
    items: List[Dict[str, Any]] = []
    gen_lens: List[int] = []

    for c in captured:
        prompt_ids: List[int] = c["prompt_token_ids"]
        gen_ids_all: List[int] = c["gen_token_ids"]
        gen_ids = gen_ids_all[: int(dump_max_new_tokens)]

        gen_lens.append(len(gen_ids))
        full_ids = prompt_ids + gen_ids
        gen_text_full = tokenizer.decode(full_ids, skip_special_tokens=True)

        items.append(
            {
                "i": int(c["i"]),
                "prompt": c["prompt"],
                "input_len": int(c["input_len"]),
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


def benchmark_pp(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    warmup: int,
    rank: int,
    world_size: int,
    pp_size: int,
    torch_dtype: torch.dtype,
    dump_n: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    def run_one(prompt: str, capture: bool = False, capture_index: int = 0) -> Tuple[int, Optional[Dict[str, Any]]]:
        # ---------- Single GPU path ----------
        if world_size == 1:
            if capture and rank == 0:
                prompt_ids, gen_ids = greedy_decode_full_model_capture(
                    model, tokenizer, prompt, max_new_tokens, device=device
                )
                cap = {
                    "i": int(capture_index),
                    "prompt": prompt,
                    "input_len": int(len(prompt_ids)),
                    "prompt_token_ids": prompt_ids,
                    "gen_token_ids": gen_ids,
                }
                return max_new_tokens, cap
            return greedy_decode_full_model(model, tokenizer, prompt, max_new_tokens, device=device), None

        # ---------- Distributed PP path (NO KV CACHE) ----------
        # 1) Rank0 tokenizes, broadcasts prompt token ids as Python list
        if rank == 0:
            enc = tokenizer(prompt, add_special_tokens=True, return_tensors=None)
            prompt_ids_list = enc["input_ids"]
            if not isinstance(prompt_ids_list, list):
                prompt_ids_list = list(prompt_ids_list)
        else:
            prompt_ids_list = None

        prompt_ids_list = _broadcast_object(prompt_ids_list, rank, world_size)
        assert isinstance(prompt_ids_list, list) and len(prompt_ids_list) > 0

        # local token buffer on each rank
        tokens: List[int] = list(prompt_ids_list)

        # Capture buffers (rank0 only)
        prompt_ids_for_capture = tokens[:] if (capture and rank == 0) else None
        gen_ids_for_capture: Optional[List[int]] = [] if (capture and rank == 0) else None

        # 2) Decode loop: each step recompute full prefix through pipeline
        for _ in range(max_new_tokens):
            seq_len = len(tokens)

            input_ids = torch.tensor(tokens, device=device, dtype=torch.long).view(1, seq_len)  # (1,S)
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).view(1, seq_len)  # (1,S)
            attn_mask_4d = make_causal_mask_4d(
                q_len=seq_len,
                k_len=seq_len,
                device=device,
                dtype=torch_dtype,   # safer than fp16 for -inf masks
            )

            # Stage pipeline forward for full prefix
            if rank == 0:
                hidden = stage_forward_nocache(
                    model=model,
                    stage_id=0,
                    pp_size=pp_size,
                    input_ids=input_ids,
                    hidden_in=None,
                    position_ids=position_ids,
                    attention_mask_4d=attn_mask_4d,
                )
                send_tensor(hidden, dst=1)
            else:
                hidden = recv_tensor((1, seq_len, model.config.hidden_size), torch_dtype, src=rank - 1, device=device)
                hidden = stage_forward_nocache(
                    model=model,
                    stage_id=rank,
                    pp_size=pp_size,
                    input_ids=None,
                    hidden_in=hidden,
                    position_ids=position_ids,
                    attention_mask_4d=attn_mask_4d,
                )
                if rank != world_size - 1:
                    send_tensor(hidden, dst=rank + 1)

            # Last stage picks next token from the LAST position logits
            if rank == world_size - 1:
                logits = last_stage_logits(model, hidden)              # (1,S,V)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1)      # (1,)
                tok = next_tok.view(1, 1).to(device)                   # (1,1)
            else:
                tok = torch.empty((1, 1), dtype=torch.long, device=device)

            dist.broadcast(tok, src=world_size - 1)
            tok_id = int(tok.item())
            tokens.append(tok_id)

            if gen_ids_for_capture is not None:
                gen_ids_for_capture.append(tok_id)

        cap: Optional[Dict[str, Any]] = None
        if capture and rank == 0:
            assert prompt_ids_for_capture is not None
            assert gen_ids_for_capture is not None
            cap = {
                "i": int(capture_index),
                "prompt": prompt,
                "input_len": int(len(prompt_ids_for_capture)),
                "prompt_token_ids": prompt_ids_for_capture,
                "gen_token_ids": gen_ids_for_capture,
            }

        return max_new_tokens, cap


    # Warmup
    for _ in range(max(0, warmup)):
        _ = run_one(prompts[0], capture=False)
        barrier_if_dist()

    barrier_if_dist()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if rank == 0:
        t0 = time.time()

    total_gen_rank0 = 0
    captured: List[Dict[str, Any]] = []

    for i, p in enumerate(prompts):
        do_cap = bool(dump_n) and (i < dump_n)
        gen, cap = run_one(p, capture=do_cap, capture_index=i)
        if rank == 0:
            total_gen_rank0 += int(gen)
            if cap is not None:
                captured.append(cap)
        barrier_if_dist()

    if rank == 0:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.time() - t0
    else:
        total_time = None

    total_time = _broadcast_object(total_time, rank, world_size)

    n = len(prompts)
    mean_latency = float(total_time) / max(1, n)
    throughput_tok_s = float(total_gen_rank0) / max(1e-9, float(total_time))

    metrics = {
        "n_prompts": n,
        "total_time_s": float(total_time),
        "mean_latency_s": float(mean_latency),
        "total_gen_tokens_rank0": int(total_gen_rank0),
        "throughput_tokens_per_s_rank0": float(throughput_tok_s),
    }
    return metrics, captured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed launcher injected arg")

    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B")

    # Keep existing arg, but add TP-matching alias.
    parser.add_argument("--pp_config", type=str, default="ds_pp.json")
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default=None,
        help="Alias for --pp_config (kept for parity with TP script).",
    )

    parser.add_argument("--benchmark_json", type=str, default=None)
    parser.add_argument("--benchmark_n", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=16)

    # Added: match TP script CLI
    parser.add_argument("--dump_outputs_json", type=str, default=None)
    parser.add_argument("--dump_n", type=int, default=5)
    parser.add_argument("--dump_max_new_tokens", type=int, default=None)

    args = parser.parse_args()

    cfg_path = args.deepspeed_config if args.deepspeed_config else args.pp_config

    rank, local_rank, world_size = init_distributed()

    if not torch.cuda.is_available():
        raise RuntimeError("This PP demo expects CUDA GPUs.")

    dtype_str, pp_size = read_pp_config(cfg_path, world_size=world_size)
    torch_dtype = _torch_dtype_from_str(dtype_str)

    dev_name = torch.cuda.get_device_name(local_rank)
    rank_print(rank, f"world_size={world_size}, local_rank={local_rank}, device=cuda:{local_rank} name={dev_name}")
    rank_print(rank, f"PP config: dtype={dtype_str}, pp_size={pp_size} (config file: {cfg_path})")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=None,
    ).to(torch.cuda.current_device())
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = len(model.model.layers)
    start, end = compute_layer_partition(num_layers, pp_size, stage_id=rank)
    rank_print(rank, f"Stage owns layers [{start}, {end}) out of {num_layers}")

    if args.benchmark_json is None:
        if rank == 0:
            print("No --benchmark_json provided. Exiting (PP script is meant to be benchmarked).", flush=True)
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    prompts = load_prompts_for_benchmark(args.benchmark_json, args.benchmark_n, rank, world_size)

    dump_n = 0
    dump_tokens = args.max_new_tokens
    if args.dump_outputs_json and rank == 0:
        dump_n = max(1, min(int(args.dump_n), len(prompts)))
        dump_tokens = int(args.dump_max_new_tokens or args.max_new_tokens)

    metrics, captured = benchmark_pp(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup,
        rank=rank,
        world_size=world_size,
        pp_size=pp_size,
        torch_dtype=torch_dtype,
        dump_n=dump_n,
    )

    if rank == 0:
        result = {
            "mode": "pp",
            "world_size": world_size,
            "pp_size": pp_size,
            "dtype": dtype_str,
            **metrics,
        }
        print("\n=== PP BENCHMARK RESULT (rank 0, JSON) ===")
        print(json.dumps(result, indent=2))
        print()

        if args.dump_outputs_json:
            print("Writing benchmark outputs (captured during timed run).")
            _finalize_and_write_dump_pp(
                out_path=args.dump_outputs_json,
                tokenizer=tokenizer,
                result_payload=result,
                captured=captured,
                dump_max_new_tokens=dump_tokens,
            )
            print(f"Wrote outputs to: {args.dump_outputs_json}", flush=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
