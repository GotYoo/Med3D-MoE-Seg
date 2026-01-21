"""DeepSpeed MoE optimizer helper.

Refactor-only extraction of the original `Med3DLISATrainer.create_optimizer`.
Logic is intentionally preserved; code is only moved out of the main entry file.
"""

from __future__ import annotations

import logging

from transformers import Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from deepspeed.moe.utils import is_moe_param, configure_moe_param_groups


def create_optimizer_moe(trainer: Trainer, logger: logging.Logger):
        """
        Setup the optimizer with MoE support.
        """
        opt_model = trainer.model

        if trainer.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            # Inspect MoE params early
            moe_params_all = [(n, p) for n, p in opt_model.named_parameters() if is_moe_param(p)]
            moe_params_rg = [(n, p) for n, p in moe_params_all if p.requires_grad]
            logger.debug(f"[Med3DLISATrainer] Detected MoE params: total={len(moe_params_all)}, requires_grad={len(moe_params_rg)}")
            if len(moe_params_rg) == 0 and len(moe_params_all) > 0:
                logger.warning("[Med3DLISATrainer] MoE params found but all frozen; unfreezing them.")
            if len(moe_params_all) > 0:
                sample_moe = [n for n, _ in moe_params_all[:5]]
                logger.debug(f"[Med3DLISATrainer] Sample MoE param names: {sample_moe}")
            else:
                logger.warning("[Med3DLISATrainer] No parameters detected by is_moe_param; DeepSpeed will fail if MoE layers exist.")
            if len(moe_params_rg) == 0:
                # Force-unfreeze MoE params so they can be optimized
                for _, p in moe_params_all:
                    p.requires_grad = True
                moe_params_rg = [(n, p) for n, p in moe_params_all if p.requires_grad]

            def collect_params(include_decay: bool, moe: bool):
                params = []
                for name, param in opt_model.named_parameters():
                    if not param.requires_grad:
                        continue
                    in_decay = name in decay_parameters
                    # Robust MoE detection: DeepSpeed flag OR name hints
                    is_moe = is_moe_param(param) or ('experts' in name) or ('deepspeed_moe' in name) or ('moe_layer' in name)
                    if include_decay != in_decay:
                        continue
                    if moe != is_moe:
                        continue
                    params.append(param)
                return params

            base_groups = []
            base_groups.append({
                "params": collect_params(include_decay=True, moe=False),
                "weight_decay": trainer.args.weight_decay,
                "name": "decay"
            })
            base_groups.append({
                "params": collect_params(include_decay=False, moe=False),
                "weight_decay": 0.0,
                "name": "no_decay"
            })
            base_groups.append({
                "params": collect_params(include_decay=True, moe=True),
                "weight_decay": trainer.args.weight_decay,
                "name": "moe_decay",
                "moe": True
            })
            base_groups.append({
                "params": collect_params(include_decay=False, moe=True),
                "weight_decay": 0.0,
                "name": "moe_no_decay",
                "moe": True
            })

            # Let DeepSpeed split MoE groups per expert-parallel group if needed
            optimizer_grouped_parameters = configure_moe_param_groups(base_groups)

            # Ensure MoE groups have expert group names DeepSpeed expects
            for g in optimizer_grouped_parameters:
                if g.get("moe", False):
                    # Try to pick the group_name from the first param
                    gn = None
                    for p in g.get("params", []):
                        if hasattr(p, "group_name") and p.group_name:
                            gn = p.group_name
                            break
                    if gn is None:
                        gn = "ep_size_1"
                    g["name"] = gn

            # Diagnostics
            total_params = sum(p.numel() for g in optimizer_grouped_parameters for p in g.get("params", []))
            moe_flagged = sum(1 for g in optimizer_grouped_parameters if g.get("moe", False))
            moe_param_count = sum(p.numel() for g in optimizer_grouped_parameters if g.get("moe", False) for p in g.get("params", []))
            moe_detected_params = sum(p.numel() for _, p in opt_model.named_parameters() if is_moe_param(p) and p.requires_grad)
            logger.debug(f"[Med3DLISATrainer] Total params in groups: {total_params}, moe-flagged groups: {moe_flagged}, moe params in groups: {moe_param_count}, detected moe params (requires_grad): {moe_detected_params}")
            # Per-group summary for visibility
            group_summaries = []
            for g in optimizer_grouped_parameters:
                group_summaries.append({
                    "name": g.get("name", ""),
                    "moe": g.get("moe", False),
                    "param_count": len(g.get("params", [])),
                    "numel": sum(p.numel() for p in g.get("params", [])),
                })
            logger.debug(f"[Med3DLISATrainer] Param groups summary: {group_summaries}")

            moe_groups = [g for g in optimizer_grouped_parameters if g.get("moe", False)]
            total_moe_params = sum(p.numel() for g in moe_groups for p in g.get("params", []))
            logger.debug(f"[Med3DLISATrainer] MoE param groups: {len(moe_groups)}, total params: {total_moe_params}")
            if not moe_groups:
                sample_names = [n for n, _ in list(opt_model.named_parameters())[:10]]
                logger.warning(f"[Med3DLISATrainer] WARNING: No MoE param groups detected. Sample names: {sample_names}")
                # Fallback: tag any params flagged by is_moe_param so DeepSpeed can proceed
                for group in optimizer_grouped_parameters:
                    if any(is_moe_param(p) or 'experts' in (getattr(p, 'name', '') or '') for p in group.get("params", [])):
                        group["moe"] = True
                moe_groups = [g for g in optimizer_grouped_parameters if g.get("moe", False)]
                total_moe_params = sum(p.numel() for g in moe_groups for p in g.get("params", []))
                logger.debug(f"[Med3DLISATrainer] After fallback, MoE groups: {len(moe_groups)}, params: {total_moe_params}")

            # Final safety: ensure at least one group is marked as MoE
            any_moe = any(g.get("moe", False) for g in optimizer_grouped_parameters)
            if not any_moe:
                # First, tag groups that actually contain MoE params
                tagged = False
                for group in optimizer_grouped_parameters:
                    if any(is_moe_param(p) for p in group.get("params", [])):
                        group["moe"] = True
                        tagged = True
                if tagged:
                    moe_groups = [g for g in optimizer_grouped_parameters if g.get("moe", False)]
                    total_moe_params = sum(p.numel() for g in moe_groups for p in g.get("params", []))
                # If still none, create an explicit MoE group with all detected MoE params
                any_moe = any(g.get("moe", False) for g in optimizer_grouped_parameters)
                if not any_moe:
                    explicit_moe_params = [p for _, p in moe_params_rg if p.requires_grad]
                    optimizer_grouped_parameters.append({
                        "params": explicit_moe_params,
                        "weight_decay": 0.0,
                        "moe": True,
                        "name": "moe_fallback_explicit"
                    })
                    moe_groups = [g for g in optimizer_grouped_parameters if g.get("moe", False)]
                    total_moe_params = sum(p.numel() for g in moe_groups for p in g.get("params", []))

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(trainer.args)
            trainer.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return trainer.optimizer
