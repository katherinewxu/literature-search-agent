"""SFT and GRPO for the small policy, sharing the prompted agent's exact turn format.

Rollouts go through the same ``LLMSearchAgent`` + environment code as evaluation,
served by vLLM with the current LoRA adapter hot-loaded; the trainer re-tokenizes
each recorded turn (chat-templated prompt + response) and computes token-level
log-probabilities with PEFT. Gold labels enter only through the episode reward.

Reward (proposal): r = |S∩R|/|R| - λ |S\\R|/|S| - c · cost/B
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .environment import LiteratureSearchEnv, allowed_ids_for
from .llm import LLMSearchAgent, OpenAIChatBackend
from .models import EpisodeSpec
from .retrieval import BM25Retriever, PooledRetrievers
from .runner import run_episode, write_run
from .store import CorpusStore


def episode_reward(metrics: dict[str, Any], budget: int, *, lam: float = 0.1, cost_weight: float = 0.0) -> float:
    included = metrics["included_count"]
    false_rate = (included - metrics["true_positives"]) / included if included else 0.0
    return metrics["recall"] - lam * false_rate - cost_weight * metrics["charged_cost"] / budget


# -- tokenization shared by SFT and GRPO -------------------------------------------
def _prompt_text(tokenizer, messages: list[dict[str, str]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def encode_turn(tokenizer, messages: list[dict[str, str]], response: str, max_length: int):
    """(input_ids, labels) with labels = -100 on the prompt; truncates the prompt's left side."""
    prompt_ids = tokenizer(_prompt_text(tokenizer, messages), add_special_tokens=False)["input_ids"]
    end = tokenizer.eos_token or "<|im_end|>"
    response_ids = tokenizer(response.strip() + end, add_special_tokens=False)["input_ids"]
    keep = max_length - len(response_ids)
    if keep <= 0:
        return None
    prompt_ids = prompt_ids[-keep:]
    return prompt_ids + response_ids, [-100] * len(prompt_ids) + response_ids


def _load_policy(config: dict[str, Any], *, adapter: str | None = None, trainable: bool = True):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(config.get("lora_rank", 32)),
                lora_alpha=int(config.get("lora_alpha", 64)),
                lora_dropout=0.0,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            ),
        )
    model.enable_input_require_grads()
    return tokenizer, model


def _token_logprobs(model, input_ids, labels):
    """Per-token log-probs of the labelled (response) positions."""
    import torch

    # Responses sit at the end of the sequence, so only the last n+1 positions need logits;
    # full-sequence fp32 logits (6k tokens x 151k vocab) do not fit beside vLLM.
    first = int((labels[0] != -100).nonzero()[0])
    keep = labels.shape[1] - first + 1
    logits = model(input_ids=input_ids, logits_to_keep=keep).logits[:, :-1, :].float()
    targets = labels[:, -(keep - 1):]
    mask = targets != -100
    logprobs = torch.log_softmax(logits, dim=-1).gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return logprobs[mask]


# -- SFT data: rejection-sampled teacher turns ---------------------------------------
def collect_sft(run_dirs: list[Path], output: Path, *, keep_fraction: float = 0.5, lam: float = 0.1) -> dict[str, Any]:
    """Keep the best-rewarded teacher episodes per (mode, budget) and write their valid turns."""
    episodes = []
    for run_dir in run_dirs:
        for path in sorted(Path(run_dir).glob("*.json")):
            run = json.loads(path.read_text(encoding="utf-8"))
            if not run.get("agent_turns"):
                continue
            summary = run.get("agent_summary") or {}
            if summary.get("parse_failures", 0) > 0:
                continue
            budget = run["episode"]["budget"]
            episodes.append((episode_reward(run["metrics"], budget, lam=lam), run))
    groups: dict[tuple[str, int], list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for reward, run in episodes:
        groups[(run["episode"]["mode"], run["episode"]["budget"])].append((reward, run))
    kept = []
    for key, items in sorted(groups.items()):
        items.sort(key=lambda item: (-item[0], item[1]["episode"]["review_id"]))
        kept.extend(items[: max(1, math.ceil(keep_fraction * len(items)))])
    output.parent.mkdir(parents=True, exist_ok=True)
    turns = 0
    with output.open("w", encoding="utf-8") as stream:
        for reward, run in kept:
            for turn in run["agent_turns"]:
                if turn.get("parsed") is None:
                    continue
                stream.write(json.dumps({
                    "review_id": run["episode"]["review_id"],
                    "mode": run["episode"]["mode"],
                    "budget": run["episode"]["budget"],
                    "reward": reward,
                    "messages": turn["messages"],
                    "response": turn["response"],
                }) + "\n")
                turns += 1
    stats = {
        "episodes_available": len(episodes),
        "episodes_kept": len(kept),
        "turns": turns,
        "reviews": len({run["episode"]["review_id"] for _, run in kept}),
        "mean_reward_kept": statistics.fmean(r for r, _ in kept) if kept else None,
        "mean_reward_all": statistics.fmean(r for r, _ in episodes) if episodes else None,
    }
    output.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def train_sft(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    rows = [json.loads(line) for line in Path(config["trajectories"]).open(encoding="utf-8")]
    tokenizer, model = _load_policy(config)
    max_length = int(config.get("max_length", 6144))
    examples = [e for row in rows if (e := encode_turn(tokenizer, row["messages"], row["response"], max_length))]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=float(config.get("learning_rate", 1e-4)))
    epochs = float(config.get("epochs", 2))
    accumulation = int(config.get("gradient_accumulation_steps", 16))
    rng = random.Random(int(config.get("seed", 329)))
    total_steps = max(1, int(len(examples) * epochs) // accumulation)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, total_steps // 20)) * max(0.0, 1 - step / total_steps)
    )
    device = next(model.parameters()).device
    model.train()
    log = []
    order: list[int] = []
    for _ in range(math.ceil(epochs)):
        epoch = list(range(len(examples)))
        rng.shuffle(epoch)
        order.extend(epoch)
    order = order[: int(len(examples) * epochs)]
    running, started = [], time.time()
    for index, example_index in enumerate(order):
        input_ids, labels = examples[example_index]
        ids = torch.tensor([input_ids], device=device)
        lab = torch.tensor([labels], device=device)
        loss = model(input_ids=ids, labels=lab).loss
        (loss / accumulation).backward()
        running.append(loss.item())
        if (index + 1) % accumulation == 0 or index + 1 == len(order):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step = (index + 1) // accumulation
            if step % 5 == 0 or index + 1 == len(order):
                entry = {"step": step, "loss": statistics.fmean(running), "elapsed": time.time() - started}
                log.append(entry)
                print(json.dumps(entry), flush=True)
                running = []
    output = Path(config["output_dir"])
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    report = {"examples": len(examples), "dropped_too_long": len(rows) - len(examples), "steps": total_steps, "log": log}
    (output / "sft_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


# -- GRPO ------------------------------------------------------------------------------
def _vllm_load_adapter(base_url: str, name: str, path: Path) -> None:
    root = base_url.rsplit("/v1", 1)[0]
    request = urllib.request.Request(
        f"{root}/v1/load_lora_adapter",
        data=json.dumps({"lora_name": name, "lora_path": str(path)}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        response.read()


def _vllm_unload_adapter(base_url: str, name: str) -> None:
    root = base_url.rsplit("/v1", 1)[0]
    request = urllib.request.Request(
        f"{root}/v1/unload_lora_adapter",
        data=json.dumps({"lora_name": name}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response.read()
    except Exception:
        pass


def train_grpo(config: dict[str, Any], base_url: str) -> dict[str, Any]:
    import torch

    store = CorpusStore.load(config["data_dir"])
    splits = json.loads((Path(config["data_dir"]) / "splits.json").read_text())
    review_ids = config.get("reviews") or splits["train"]
    modes = config.get("modes", ["hybrid"])
    budgets = config.get("budgets", [20])
    group_size = int(config.get("group_size", 8))
    prompts_per_step = int(config.get("prompts_per_step", 4))
    steps = int(config.get("steps", 40))
    lam = float(config.get("lambda", 0.1))
    cost_weight = float(config.get("cost_weight", 0.0))
    beta = float(config.get("kl_beta", 0.02))
    temperature = float(config.get("temperature", 1.0))
    max_length = int(config.get("max_length", 6144))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(config.get("seed", 329)))

    # Policy = SFT adapter (trainable); reference = the same adapter frozen at step 0.
    tokenizer, model = _load_policy(config, adapter=config.get("init_adapter"), trainable=True)
    reference = None
    if beta > 0:
        _, reference = _load_policy(config, adapter=config.get("init_adapter"), trainable=False)
        reference.eval()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=float(config.get("learning_rate", 2e-6)))
    device = next(model.parameters()).device
    pooled = PooledRetrievers(BM25Retriever)
    nothink = {"chat_template_kwargs": {"enable_thinking": False}}

    def publish(step: int) -> str:
        path = output / f"adapter-step{step:03d}"
        model.save_pretrained(path)
        name = f"policy-{step:03d}"
        _vllm_load_adapter(base_url, name, path)
        return name

    adapter_name = publish(0)
    history = []
    for step in range(steps):
        started = time.time()
        tasks = [(rng.choice(review_ids), rng.choice(modes), rng.choice(budgets)) for _ in range(prompts_per_step)]
        backend = OpenAIChatBackend(adapter_name, base_url=base_url, max_tokens=int(config.get("max_tokens", 600)), extra_body=nothink)

        def rollout(job: tuple[int, int]) -> dict[str, Any]:
            task_index, member = job
            review_id, mode, budget = tasks[task_index]
            review = store.review(review_id)
            retriever = pooled.for_pool(review_id, (store.papers[i] for i in sorted(allowed_ids_for(store, review))))
            spec = EpisodeSpec(review.id, review.question, review.eligibility_criteria, budget, mode)
            agent = LLMSearchAgent(backend, name="grpo", temperature=temperature, seed=rng_seed + task_index * 1000 + member)
            run = run_episode(LiteratureSearchEnv(store, retriever, spec), agent)
            run["reward"] = episode_reward(run["metrics"], budget, lam=lam, cost_weight=cost_weight)
            run["group"] = task_index
            return run

        rng_seed = rng.randrange(1 << 30)
        jobs = [(t, g) for t in range(len(tasks)) for g in range(group_size)]
        with ThreadPoolExecutor(max_workers=int(config.get("rollout_workers", 32))) as pool:
            runs = list(pool.map(rollout, jobs))
        rollout_seconds = time.time() - started

        # Group-normalized advantages; groups with no reward variance carry no signal.
        by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for run in runs:
            by_group[run["group"]].append(run)
        samples = []
        for group in by_group.values():
            rewards = [run["reward"] for run in group]
            spread = statistics.pstdev(rewards)
            if spread < 1e-6:
                continue
            mean = statistics.fmean(rewards)
            for run in group:
                advantage = (run["reward"] - mean) / (spread + 1e-4)
                for turn in run["agent_turns"]:
                    encoded = encode_turn(tokenizer, turn["messages"], turn["response"], max_length)
                    if encoded:
                        samples.append((advantage, encoded))
        total_tokens = sum(sum(1 for t in labels if t != -100) for _, (_, labels) in samples) or 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        kl_sum, pg_sum = 0.0, 0.0
        for advantage, (input_ids, labels) in samples:
            ids = torch.tensor([input_ids], device=device)
            lab = torch.tensor([labels], device=device)
            logprobs = _token_logprobs(model, ids, lab)
            # On-policy single update: ratio is 1 at the sample point, so the clipped
            # GRPO surrogate reduces to the advantage-weighted log-likelihood gradient.
            loss = -(advantage * logprobs).sum()
            if reference is not None:
                with torch.no_grad():
                    ref_logprobs = _token_logprobs(reference, ids, lab)
                log_ratio = ref_logprobs - logprobs
                kl = (torch.exp(log_ratio) - log_ratio - 1).sum()
                loss = loss + beta * kl
                kl_sum += kl.item()
            pg_sum += -(advantage * logprobs.detach()).sum().item()
            (loss / total_tokens).backward()
        if samples:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        old_adapter = adapter_name
        adapter_name = publish(step + 1)
        _vllm_unload_adapter(base_url, old_adapter)
        entry = {
            "step": step + 1,
            "tasks": tasks,
            "mean_reward": statistics.fmean(run["reward"] for run in runs),
            "mean_recall": statistics.fmean(run["metrics"]["recall"] for run in runs),
            "mean_precision": statistics.fmean(run["metrics"]["precision"] for run in runs),
            "parse_failures": sum((run.get("agent_summary") or {}).get("parse_failures", 0) for run in runs),
            "informative_groups": sum(1 for g in by_group.values() if statistics.pstdev([r["reward"] for r in g]) >= 1e-6),
            "train_turns": len(samples),
            "kl_per_token": kl_sum / total_tokens,
            "pg_per_token": pg_sum / total_tokens,
            "rollout_seconds": rollout_seconds,
            "step_seconds": time.time() - started,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        with (output / "grpo_log.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        if config.get("save_rollouts"):
            for index, run in enumerate(runs):
                write_run(run, output / "rollouts" / f"step{step + 1:03d}_{index:03d}.json")
    model.save_pretrained(output / "final")
    tokenizer.save_pretrained(output / "final")
    return {"steps": steps, "history": history}
