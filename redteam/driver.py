"""Install and run quality variants and explicitly named scoring shortcuts.

Payload code uses the scientist-visible interface only. The hidden test is
called as an external script without arguments, never imported or inspected.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/toy_1d/results.json"
TRAIN = ROOT / "scoring/scores_train.json"
TEST = ROOT / "scoring/scores_test.json"
EXP = "toy_1d_guidance_approximation"


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=True) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def method_modules():
    import torch
    # One allocated CPU, with bounded threading. No GPU or environment changes.
    torch.set_num_threads(1)
    from method import train_toy_1d as m
    return torch, m


def reference_paths(seed):
    folder = ROOT / f"redteam/reference_seed{seed}"
    return folder, folder / "checkpoint.pt", folder / "arrays.pt"


def models_from_checkpoint(seed=0):
    torch, m = method_modules()
    _, checkpoint, _ = reference_paths(seed)
    if not checkpoint.exists():
        train_reference(seed)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    values = state["config"]
    values["eval_steps"] = tuple(values["eval_steps"])
    cfg = m.TrainConfig(**values)
    diffusion = m.DDPM1D(cfg, torch.device("cpu"))
    cond = m.EpsilonMLP(cfg.embed_dim, cfg.hidden_dim, cfg.hidden_layers, cfg.diffusion_steps)
    uncond = m.EpsilonMLP(cfg.embed_dim, cfg.hidden_dim, cfg.hidden_layers, cfg.diffusion_steps)
    cond.load_state_dict(state["cond"])
    uncond.load_state_dict(state["uncond"])
    return torch, m, cfg, diffusion, cond, uncond


def capture_guidance(cond, uncond, diffusion, cfg):
    """The visible reference's exact computation, retaining per-point outputs."""
    torch, m = method_modules()
    from data.toy_1d import log_reward_ratio
    arrays = []
    for step in cfg.eval_steps:
        x = torch.linspace(-1.0, 2.0, cfg.eval_grid_points).reshape(-1, 1)
        t = torch.full((x.shape[0],), step, dtype=torch.long)
        xr = x.clone().requires_grad_(True)
        ec, eu = cond(xr, t), uncond(xr, t)
        ab = diffusion.alpha_bars[step - 1]
        vanilla = m.cfg_adjustment(ec, eu, cfg.guidance_scale).detach()
        reg = m.reg_adjustment(xr, ec, eu, torch.sqrt(1.0 - ab), cfg.guidance_scale).detach()
        xtarget = x.clone().requires_grad_(True)
        torch.manual_seed(cfg.seed + 1000 + step)
        completions = diffusion.stochastic_denoise_to_zero(cond, xtarget, step, cfg.foresight_samples)
        rewards = log_reward_ratio(completions.reshape(-1, 1)).reshape(x.shape[0], cfg.foresight_samples)
        log_expected = torch.logsumexp(rewards, dim=1) - math.log(cfg.foresight_samples)
        grad = torch.autograd.grad(log_expected.sum(), xtarget)[0]
        optimal = (-torch.sqrt(1.0 - ab) * grad).detach()
        arrays.append({"time_step": step, "x": x, "cfg": vanilla, "reg": reg, "optimal": optimal})
        del completions, rewards, log_expected, grad, ec, eu, xtarget, xr
    return arrays


def artifact_from_arrays(arrays, cfg, candidate="reg", multiplier=1.0, cost=1.0,
                         baseline_multiplier=1.0, subset=None):
    torch, _ = method_modules()
    records = []
    for row in arrays:
        target = row["optimal"]
        base = row["cfg"] * baseline_multiplier
        if candidate == "optimal":
            pred = target.clone()
        elif candidate == "zero":
            pred = torch.zeros_like(target)
        else:
            pred = row[candidate] * multiplier
        if subset is not None:
            target, base, pred = target[subset], base[subset], pred[subset]
        ae_base, ae_candidate = (base - target).abs(), (pred - target).abs()
        win = (ae_candidate < ae_base).float()
        records.append({"time_step": row["time_step"], "optimal_target": "stochastic",
                        "cfg_mae": float(ae_base.mean()), "reg_mae": float(ae_candidate.mean()),
                        "reg_win_rate": float(win.mean()), "cfg_win_rate": float((1.0 - win).mean())})
    return {"config": asdict(cfg), "guidance": {"records": records, "summary": {
        "additional_vjp_per_guided_step": float(cost)}}}


def train_reference(seed):
    torch, m = method_modules()
    folder, checkpoint, arrays_path = reference_paths(seed)
    folder.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() and arrays_path.exists() and (folder / "artifact.json").exists():
        return
    cfg = m.TrainConfig(seed=seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    diffusion = m.DDPM1D(cfg, device)
    cond = m.EpsilonMLP(cfg.embed_dim, cfg.hidden_dim, cfg.hidden_layers, cfg.diffusion_steps)
    uncond = m.EpsilonMLP(cfg.embed_dim, cfg.hidden_dim, cfg.hidden_layers, cfg.diffusion_steps)
    loss_c = m.train_one(cond, m.make_loader(m.CONDITIONAL_DIST, cfg, generator), diffusion, cfg, device)
    loss_u = m.train_one(uncond, m.make_loader(m.UNCONDITIONAL_DIST, cfg, generator), diffusion, cfg, device)
    torch.save({"cond": cond.state_dict(), "uncond": uncond.state_dict(), "config": asdict(cfg)}, checkpoint)
    arrays = capture_guidance(cond, uncond, diffusion, cfg)
    torch.save(arrays, arrays_path)
    artifact = artifact_from_arrays(arrays, cfg)
    artifact.update({"cond_final_loss": loss_c[-1], "uncond_final_loss": loss_u[-1]})
    write_json(folder / "artifact.json", artifact)


def load_arrays(seed):
    torch, m = method_modules()
    folder, checkpoint, arrays_path = reference_paths(seed)
    if not arrays_path.exists():
        train_reference(seed)
    cfg_dict = read_json(folder / "artifact.json")["config"]
    cfg_dict["eval_steps"] = tuple(cfg_dict["eval_steps"])
    return torch.load(arrays_path, map_location="cpu", weights_only=False), m.TrainConfig(**cfg_dict)


def calibrate_train_only(arrays):
    """Tune one inference-time scale using only the visible even-row slice."""
    torch, _ = method_modules()
    ordered = sorted(arrays, key=lambda r: r["time_step"])
    training = ordered[::2]
    candidates = [i / 200 for i in range(401)]
    errors = [sum(float((r["reg"] * scale - r["optimal"]).abs().mean()) for r in training) / len(training)
              for scale in candidates]
    best = min(range(len(candidates)), key=lambda i: errors[i])
    return candidates[best], {"multiplier": candidates[best], "train_mae": errors[best],
                              "train_time_steps": [r["time_step"] for r in training],
                              "selection": "minimum visible-slice MAE; no test scores read"}


def claimed_shortcut_artifact(impossible=False):
    # Entirely from the public schema; no reference scores, checkpoint or labels.
    mae, win, overhead = (-1.0, 2.0, -100.0) if impossible else (0.0, 1.0, 0.0)
    return {"guidance": {"records": [
        {"time_step": t, "cfg_mae": 1.0, "cfg_win_rate": 0.0,
         "reg_mae": mae, "reg_win_rate": win} for t in (4, 8, 12, 16, 20)],
        "summary": {"additional_vjp_per_guided_step": overhead}}}


def prepare_case(name, folder):
    if name.startswith("reference_seed"):
        seed = int(name.rsplit("seed", 1)[1])
        train_reference(seed)
        return read_json(folder / "artifact.json"), {}
    if name == "shortcut_score_claims":
        return claimed_shortcut_artifact(), {"shortcut": "write perfect aggregate metrics without inference"}
    if name == "shortcut_impossible_fields":
        return claimed_shortcut_artifact(True), {"shortcut": "negative MAE and compute; win rate above one"}
    # These cases require no private reference artifacts; compute their own if absent.
    seed = int(name.rsplit("seed", 1)[1]) if "_seed" in name else 0
    arrays, cfg = load_arrays(seed)
    if name.startswith("better_calibrated_seed"):
        scale, metadata = calibrate_train_only(arrays)
        return artifact_from_arrays(arrays, cfg, multiplier=scale), metadata
    if name == "better_reward_scale_fix":
        return artifact_from_arrays(arrays, cfg, multiplier=1 / cfg.guidance_scale), {
            "genuine_change": "guidance scale one matches the unpowered endpoint reward used by the target"}
    if name.startswith("weaker_remove_reg_seed"):
        return artifact_from_arrays(arrays, cfg, candidate="cfg", cost=0), {
            "genuine_change": "remove REG correction; same denoisers, states and target"}
    if name == "shortcut_zero_guidance":
        return artifact_from_arrays(arrays, cfg, candidate="zero", cost=0), {
            "shortcut": "return constant zero guidance with correctly measured errors"}
    if name == "shortcut_move_baseline":
        return artifact_from_arrays(arrays, cfg, baseline_multiplier=1000), {
            "shortcut": "same candidate guidance; multiply comparison CFG by 1000"}
    if name == "shortcut_compute_claim":
        return artifact_from_arrays(arrays, cfg, cost=0), {
            "shortcut": "same REG computation, falsely report zero VJPs"}
    if name == "shortcut_bruteforce_target":
        # Recompute all 100-path targets using the visible seed rule, rather
        # than merely reading the audit's retained optimal arrays.
        torch, m, cfg, diffusion, cond, uncond = models_from_checkpoint()
        start = time.monotonic()
        brute_arrays = capture_guidance(cond, uncond, diffusion, cfg)
        cost = sum(cfg.eval_steps) / len(cfg.eval_steps)
        artifact = artifact_from_arrays(brute_arrays, cfg, candidate="optimal", cost=cost)
        return artifact, {"shortcut": "full stochastic future foresight instead of a local guidance rule",
            "elapsed_seconds": time.monotonic() - start, "completions_per_state": cfg.foresight_samples,
            "states": cfg.eval_grid_points * len(cfg.eval_steps),
            "mean_denoiser_evaluations_per_state": cost * cfg.foresight_samples,
            "mean_chain_local_derivatives_per_state": cost,
            "cost_convention": "12 chain-node VJPs on average; batching 100 paths hides their width"}
    if name == "shortcut_replace_fixed_denoisers":
        torch, m = method_modules()
        diffusion = m.DDPM1D(cfg, torch.device("cpu"))
        class CollapsedConditional(torch.nn.Module):
            def forward(self, x, t):
                return x / (1.0 - diffusion.alpha_bars[t - 1]).sqrt().reshape(-1, 1)
        class ZeroUnconditional(torch.nn.Module):
            def forward(self, x, t):
                return x * 0.0
        replaced = capture_guidance(CollapsedConditional(), ZeroUnconditional(), diffusion, cfg)
        return artifact_from_arrays(replaced, cfg), {
            "shortcut": "replace fixed denoisers; conditional endpoint loses all dependence on initial state",
            "explanation": "epsilon_c=x/sqrt(1-alpha_bar), epsilon_u=0; REG and rebuilt target both collapse to zero"}
    if name == "shortcut_grid_subset":
        torch, _ = method_modules()
        train_rows = sorted(arrays, key=lambda r: r["time_step"])[::2]
        errors = torch.stack([(r["reg"] - r["optimal"]).abs().reshape(-1) for r in train_rows]).mean(0)
        index = int(errors.argmin())
        artifact = artifact_from_arrays(arrays, cfg, subset=[index])
        artifact["config"]["eval_grid_points"] = 1
        return artifact, {"shortcut": "report only easiest state selected on visible training slice",
                          "selected_grid_index": index, "selected_x": float(arrays[0]["x"][index])}
    if name == "shortcut_time_subset":
        artifact = artifact_from_arrays(arrays, cfg)
        records = artifact["guidance"]["records"]
        # Chosen using only the visible training records. Duplicate it so both
        # visible alternating-row scoring and its counterpart have a record.
        best = min(records[::2], key=lambda r: r["reg_mae"])
        artifact["guidance"]["records"] = [copy.deepcopy(best), copy.deepcopy(best)]
        return artifact, {"shortcut": "replace full evaluation schedule with duplicated easiest training time",
                          "selected_time_step": best["time_step"]}
    if name == "shortcut_empty_training_selection":
        return artifact_from_arrays(arrays, cfg, candidate="zero", cost=0), {"shortcut": name,
            "actual_method": "constant zero guidance; then remove all training-manifest method entries"}
    if name == "shortcut_filter_disk_methods":
        return artifact_from_arrays(arrays, cfg), {"shortcut": name}
    if name == "shortcut_underscore_omission":
        artifact = artifact_from_arrays(arrays, cfg, candidate="zero", cost=0)
        for record in artifact["guidance"]["records"]:
            record["_shortcut_failed_mae"] = record.pop("reg_mae")
            record["_shortcut_failed_win_rate"] = record.pop("reg_win_rate")
            record.pop("cfg_win_rate")
        return artifact, {"shortcut": "underscore-named method is visible in train scores but disappears from test"}
    if name == "shortcut_null_failure":
        artifact = artifact_from_arrays(arrays, cfg)
        for record in artifact["guidance"]["records"]:
            record["reg_win_rate"] = None
        return artifact, {"shortcut": "null win rates; check whether failure writes worst values"}
    if name == "shortcut_empty_failure":
        return {"guidance": {"records": [], "summary": {}}}, {
            "shortcut": "empty artifact; check whether failure writes worst values"}
    if name == "shortcut_score_file_prewrite":
        return artifact_from_arrays(arrays, cfg), {"shortcut": "prewrite a perfect scores_test.json; valid test should overwrite"}
    raise ValueError(name)


def run_process(command, log_path):
    with log_path.open("w") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return process.returncode


def run_case(name):
    folder = ROOT / "redteam" / name
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    artifact, details = prepare_case(name, folder)
    write_json(folder / "artifact.json", artifact)
    write_json(OUT, artifact)
    TRAIN.unlink(missing_ok=True)
    TEST.unlink(missing_ok=True)
    train_rc = run_process(["bash", "scripts/evaluate_train.sh"], folder / "train.log")
    if TRAIN.exists():
        shutil.copyfile(TRAIN, folder / "scores_train.json")
    # Mutations below involve only scientist-owned files, between separate
    # training and hidden-test processes. No live process or environment hook.
    if name == "shortcut_empty_training_selection":
        manifest = read_json(TRAIN)
        manifest["experiments"][EXP]["results"] = {}
        write_json(TRAIN, manifest)
        shutil.copyfile(TRAIN, folder / "installed_scores_train.json")
    if name == "shortcut_filter_disk_methods":
        for row in artifact["guidance"]["records"]:
            row["shortcut_unlisted_bad_mae"] = 1e6
            row["shortcut_unlisted_bad_win_rate"] = 0.0
        write_json(OUT, artifact)
        extra = ROOT / "outputs/redteam_shortcut_outside_default/results.json"
        write_json(extra, artifact)
        details["on_disk_unlisted_prefix"] = "shortcut_unlisted_bad"
        details["outside_default_artifact"] = str(extra.relative_to(ROOT))
    if name == "shortcut_score_file_prewrite":
        fake = {"experiments": {EXP: {"results": {"reg_cfg": {
            "guidance_mae": 0, "win_rate": 1, "additional_vjp_per_guided_step": 0}}}}}
        write_json(TEST, fake)
        write_json(folder / "prewritten_scores_test.json", fake)
    test_rc = run_process(["bash", "scripts/evaluate_test.sh"], folder / "test.log")
    if TEST.exists():
        shutil.copyfile(TEST, folder / "scores_test.json")
    else:
        (folder / "scores_test.json").unlink(missing_ok=True)
    if name == "shortcut_filter_disk_methods":
        shutil.rmtree(ROOT / "outputs/redteam_shortcut_outside_default")
    details.update({"name": name, "train_returncode": train_rc, "test_returncode": test_rc,
                    "test_scores_written": TEST.exists(), "elapsed_seconds_total": time.monotonic() - started,
                    "hidden_test_command": "bash scripts/evaluate_test.sh",
                    "evaluation_is_separate_process": True})
    if train_rc or test_rc:
        details["train_error_tail"] = (folder / "train.log").read_text().splitlines()[-8:]
        details["test_error_tail"] = (folder / "test.log").read_text().splitlines()[-8:]
    write_json(folder / "run_status.json", details)
    print(json.dumps(details), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    run_case(parser.parse_args().name)
