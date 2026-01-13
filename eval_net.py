"""
Evaluation entry for Med3D-MoE-Seg.
Current focus: Stage 1 multi-modal alignment checkpoint evaluation.
Outputs metrics JSON under the provided output directory.
"""

import argparse
import json
import os
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
	import yaml
except Exception:
	yaml = None

from data.alignment_dataset import LIDCAlignmentDataset, alignment_collate_fn
from model.alignment import AlignmentModel


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate Med3D-MoE-Seg")
	parser.add_argument("--config", type=str, default="config/stage1_alignment.yaml")
	parser.add_argument("--checkpoint", type=str, required=True)
	parser.add_argument("--stage", type=str, default="stage1_alignment")
	parser.add_argument("--output_dir", type=str, default="eval_results/stage1_alignment")
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--batch_size", type=int, default=2)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--test_json", type=str, default=None, help="override test split json")
	parser.add_argument("--data_root", type=str, default=None, help="override data root")
	parser.add_argument("--max_batches", type=int, default=None, help="debug: limit batches")
	return parser.parse_args()


def load_config(config_path: Optional[str]) -> Dict:
	if not config_path:
		return {}

	cfg_path = Path(config_path)
	if not cfg_path.exists():
		raise FileNotFoundError(f"Config not found: {config_path}")

	if cfg_path.suffix in {".yml", ".yaml"}:
		if yaml is None:
			raise ImportError("pyyaml is required to load YAML configs")
		with open(cfg_path, "r", encoding="utf-8") as f:
			return yaml.safe_load(f)

	with open(cfg_path, "r", encoding="utf-8") as f:
		return json.load(f)


def build_stage1_model(config: Dict) -> AlignmentModel:
	model_cfg = config.get("model", {}) if config else {}
	return AlignmentModel(
		ct_clip_config=model_cfg.get("ct_clip_encoder", {}),
		pixel_config=model_cfg.get("pixel_encoder", {}),
		text_config=model_cfg.get("text_encoder", {}),
		alignment_config=model_cfg.get("alignment", {}),
	)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
	if not os.path.isfile(checkpoint_path):
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	# Prefer loading weights only to avoid unpickling old classes
	try:
		ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)  # type: ignore[arg-type]
		state_dict = ckpt
	except TypeError:
		ckpt = torch.load(checkpoint_path, map_location=device)
		state_dict = ckpt.get("state_dict", ckpt)

	cleaned = {}
	for k, v in state_dict.items():
		if k.startswith("module."):
			cleaned[k[7:]] = v
		else:
			cleaned[k] = v
	missing, unexpected = model.load_state_dict(cleaned, strict=False)

	if missing:
		print(f"[WARN] Missing keys when loading checkpoint: {missing}")
	if unexpected:
		print(f"[WARN] Unexpected keys in checkpoint: {unexpected}")


def _extract_stage1_data_config(config: Dict, args: argparse.Namespace) -> Dict:
	if config is None:
		return {}

	if "data" in config:
		return config.get("data", {})

	stages = config.get("training_stages", {})
	if isinstance(stages, dict):
		if "stage1_alignment" in stages:
			stage_cfg = stages["stage1_alignment"]
		elif "stage1" in stages:
			stage_cfg = stages["stage1"]
		else:
			stage_cfg = None

		if stage_cfg is not None:
			data_root = stage_cfg.get("data_root")
			params = stage_cfg.get("dataset_params", {})
			image_size = params.get("image_size", stage_cfg.get("image_size", [96, 96, 96]))
			num_slices = params.get("num_slices", stage_cfg.get("num_slices", 8))
			return {
				"data_root": data_root,
				"train_json": stage_cfg.get("train_source"),
				"val_json": stage_cfg.get("val_source"),
				"test_json": stage_cfg.get("test_source"),
				"image_size": image_size,
				"num_slices": num_slices,
			}

	return {}


def prepare_stage1_dataloader(config: Dict, args: argparse.Namespace, tokenizer) -> DataLoader:
	data_cfg = _extract_stage1_data_config(config, args)

	data_root = args.data_root or data_cfg.get("data_root")
	if data_root is None:
		raise ValueError("data_root is required for stage1 evaluation")

	annotation = args.test_json or data_cfg.get("test_json") or data_cfg.get("val_json")
	if annotation is None:
		raise ValueError("test_json or val_json must be provided for evaluation")
	if not Path(annotation).exists():
		raise FileNotFoundError(f"Annotation JSON not found: {annotation}")

	image_size = tuple(data_cfg.get("image_size", [96, 96, 96]))
	num_slices = int(data_cfg.get("num_slices", 8))

	dataset = LIDCAlignmentDataset(
		data_root=data_root,
		annotation_file=annotation,
		image_size=image_size,
		num_slices=num_slices,
	)

	collate = partial(alignment_collate_fn, tokenizer=tokenizer)
	return DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		collate_fn=collate,
		pin_memory=True,
	)


def _stack_embeddings(buf: List[torch.Tensor]) -> Optional[torch.Tensor]:
	if not buf:
		return None
	return torch.cat(buf, dim=0)


def _recall_at_k(sim_matrix: torch.Tensor, ks: List[int]) -> Dict[str, float]:
	if sim_matrix is None:
		return {}

	results: Dict[str, float] = {}
	target = torch.arange(sim_matrix.size(0))
	for k in ks:
		k_eff = min(k, sim_matrix.size(1))
		topk = sim_matrix.topk(k_eff, dim=1).indices
		hits = (topk == target.unsqueeze(1)).any(dim=1).float()
		results[f"r@{k}"] = float(hits.mean().item())
	return results


def evaluate_stage1(model: AlignmentModel, dataloader: DataLoader, device: torch.device,
					max_batches: Optional[int] = None) -> Dict[str, float]:
	model.eval()
	losses = []
	ct_text_losses = []
	pixel_text_losses = []
	ct_pixel_losses = []

	ct_embeds: List[torch.Tensor] = []
	pixel_embeds: List[torch.Tensor] = []
	text_embeds: List[torch.Tensor] = []

	with torch.no_grad():
		for step, batch in enumerate(tqdm(dataloader, desc="Evaluating", ncols=100)):
			if max_batches is not None and step >= max_batches:
				break

			ct_volume = batch["ct_volume"].to(device)
			ct_slices = batch["ct_slices"].to(device)
			text_inputs = {k: v.to(device) for k, v in batch["text_inputs"].items()}

			outputs = model(
				ct_volume=ct_volume,
				ct_slices=ct_slices,
				text_inputs=text_inputs,
				return_embeddings=True,
			)

			loss = outputs.get("loss")
			if loss is not None:
				losses.append(float(loss.detach().cpu().item()))

			if "ct_clip_text_loss" in outputs:
				ct_text_losses.append(float(outputs["ct_clip_text_loss"].detach().cpu().item()))
			if "pixel_text_loss" in outputs:
				pixel_text_losses.append(float(outputs["pixel_text_loss"].detach().cpu().item()))
			if "ct_clip_pixel_loss" in outputs:
				ct_pixel_losses.append(float(outputs["ct_clip_pixel_loss"].detach().cpu().item()))

			emb = outputs.get("embeddings", {})
			if emb.get("ct_clip") is not None:
				ct_embeds.append(emb["ct_clip"].detach().cpu())
			if emb.get("pixel") is not None:
				pixel_embeds.append(emb["pixel"].detach().cpu())
			if emb.get("text") is not None:
				text_embeds.append(emb["text"].detach().cpu())

	# Aggregate embeddings on CPU to avoid extra GPU memory
	ct_all = _stack_embeddings(ct_embeds)
	pixel_all = _stack_embeddings(pixel_embeds)
	text_all = _stack_embeddings(text_embeds)

	metrics: Dict[str, float] = {}

	if losses:
		metrics["loss"] = float(sum(losses) / len(losses))
	if ct_text_losses:
		metrics["ct_clip_text_loss"] = float(sum(ct_text_losses) / len(ct_text_losses))
	if pixel_text_losses:
		metrics["pixel_text_loss"] = float(sum(pixel_text_losses) / len(pixel_text_losses))
	if ct_pixel_losses:
		metrics["ct_clip_pixel_loss"] = float(sum(ct_pixel_losses) / len(ct_pixel_losses))

	ks = [1, 3, 5]

	def _normalize(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
		return F.normalize(x, dim=-1) if x is not None else None

	ct_norm = _normalize(ct_all)
	text_norm = _normalize(text_all)
	pixel_norm = _normalize(pixel_all)

	if ct_norm is not None and text_norm is not None:
		sim_ct_text = ct_norm @ text_norm.t()
		i2t = _recall_at_k(sim_ct_text, ks)
		t2i = _recall_at_k(sim_ct_text.t(), ks)
		metrics.update({f"ct_i2t_{k}": v for k, v in i2t.items()})
		metrics.update({f"ct_t2i_{k}": v for k, v in t2i.items()})
		if "r@1" in i2t and "r@1" in t2i:
			metrics["alignment_accuracy"] = float((i2t["r@1"] + t2i["r@1"]) / 2.0)

	if pixel_norm is not None and text_norm is not None:
		sim_px_text = pixel_norm @ text_norm.t()
		i2t_px = _recall_at_k(sim_px_text, ks)
		t2i_px = _recall_at_k(sim_px_text.t(), ks)
		metrics.update({f"pixel_i2t_{k}": v for k, v in i2t_px.items()})
		metrics.update({f"pixel_t2i_{k}": v for k, v in t2i_px.items()})

	def _num_samples() -> int:
		for tensor in (ct_all, pixel_all, text_all):
			if tensor is not None:
				return int(tensor.size(0))
		return 0

	metrics["num_samples"] = _num_samples()
	return metrics


def main():
	args = parse_args()

	cfg = load_config(args.config)

	stage = args.stage.lower()
	if stage not in {"stage1", "stage1_alignment", "alignment"}:
		raise NotImplementedError(f"Only stage1 alignment evaluation is implemented, got {args.stage}")

	stage_key = "stage1_alignment"

	device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

	print(f"Using device: {device}")
	print(f"Loading config: {args.config}")
	print(f"Checkpoint: {args.checkpoint}")

	model = build_stage1_model(cfg).to(device)
	load_checkpoint(model, args.checkpoint, device)

	tokenizer = model.get_text_tokenizer()
	dataloader = prepare_stage1_dataloader(cfg, args, tokenizer)

	metrics = evaluate_stage1(model, dataloader, device, max_batches=args.max_batches)

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	metrics_path = output_dir / f"{stage_key}_metrics.json"
	with open(metrics_path, "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)

	print("Evaluation completed.")
	print(json.dumps(metrics, indent=2))
	print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
	main()
