import os
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pylidc as pl
import nibabel as nib
from tqdm import tqdm
from pylidc.utils import consensus


# =========================
# 配置
# =========================
@dataclass
class Config:
    output_dir: str = "datasets/LIDC-IDRI/processed/LIDC"
    consensus_level: float = 0.5
    pad: int = 1

    keep_negative_scans: bool = True
    overwrite: bool = False
    validate_existing_files: bool = True

    log_path: str = "lidc_preprocess.log"


cfg = Config()

OUTPUT_DIR = cfg.output_dir
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
MASK_DIR  = os.path.join(OUTPUT_DIR, "masks")
META_DIR  = os.path.join(OUTPUT_DIR, "meta")
INDEX_JSON = os.path.join(OUTPUT_DIR, "lidc_dataset.json")

for d in (IMAGE_DIR, MASK_DIR, META_DIR):
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=cfg.log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================
# ✅ pylidc 版本兼容关键函数
# =========================
def get_clusters(scan):
    """
    兼容 pylidc 不同版本：
    - scan.cluster_annotations 是 method
    - scan.cluster_annotations 是 list
    """
    ca = getattr(scan, "cluster_annotations", None)
    if ca is None:
        return []
    return ca() if callable(ca) else ca


# =========================
# spacing & affine
# =========================
def get_spacing_xyz(scan):
    ps = scan.pixel_spacing
    if isinstance(ps, (list, tuple, np.ndarray)):
        if len(ps) >= 2:
            sy = float(ps[0])
            sx = float(ps[1])
        else:
            sx = sy = float(ps[0])
    else:
        sx = sy = float(ps)
    sz = float(scan.slice_spacing)
    return (sx, sy, sz)


def make_affine_from_spacing(spacing_xyz):
    sx, sy, sz = spacing_xyz
    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = sx
    affine[1, 1] = sy
    affine[2, 2] = sz
    return affine


# =========================
# 报告文本
# =========================
def generate_text_report(cluster_stats):
    if len(cluster_stats) == 0:
        return "No nodules detected in this scan."

    primary = max(cluster_stats, key=lambda x: x["malignancy_avg"])
    n = len(cluster_stats)

    mal = primary["malignancy_avg"]
    tex = primary["texture_avg"]
    sp  = primary["spiculation_avg"]

    if mal >= 4:
        mal_desc = "a highly suspicious malignant nodule"
        impression = "suspicious"
    elif mal >= 3:
        mal_desc = "an indeterminate nodule"
        impression = "indeterminate"
    else:
        mal_desc = "a likely benign nodule"
        impression = "likely benign"

    if tex <= 2:
        tex_desc = "ground-glass opacity"
    elif tex >= 4:
        tex_desc = "solid"
    else:
        tex_desc = "semi-solid"

    if sp <= 2:
        sp_desc = "spiculated margin"
    elif sp <= 3.5:
        sp_desc = "mildly irregular margin"
    else:
        sp_desc = "smooth margin"

    vox = primary["voxels"]

    return (
        f"CT scan with {n} annotated nodule cluster(s). "
        f"Primary finding: {mal_desc} with {tex_desc} appearance and {sp_desc}. "
        f"Overall impression: {impression}. "
        f"(Primary nodule mask volume: {vox} voxels.)"
    )


# =========================
# mask + cluster stats
# =========================
def build_mask_and_cluster_stats(scan, vol_shape):
    mask_vol = np.zeros(vol_shape, dtype=np.uint8)
    cluster_stats = []

    clusters = get_clusters(scan)
    if len(clusters) == 0:
        return mask_vol, cluster_stats

    for ci, cluster in enumerate(clusters):
        c_mask, c_bbox, _ = consensus(
            cluster,
            clevel=cfg.consensus_level,
            pad=cfg.pad
        )

        sub = mask_vol[c_bbox]

        # === 安全检查：防 pylidc consensus 异常 ===
        if sub.shape != c_mask.shape:
            # 记录但跳过这个 cluster
            logging.warning(
                f"Skipping malformed cluster: bbox shape {sub.shape} "
                f"!= mask shape {c_mask.shape}"
            )
            continue

        sub[c_mask.astype(bool)] = 1
        mask_vol[c_bbox] = sub

        cluster_stats.append({
            "cluster_id": int(ci),
            "voxels": int(c_mask.sum()),
            "bbox": {
                "x": [int(c_bbox[0].start), int(c_bbox[0].stop)],
                "y": [int(c_bbox[1].start), int(c_bbox[1].stop)],
                "z": [int(c_bbox[2].start), int(c_bbox[2].stop)],
            },
            "malignancy_avg": float(np.mean([a.malignancy for a in cluster])),
            "spiculation_avg": float(np.mean([a.spiculation for a in cluster])),
            "texture_avg": float(np.mean([a.texture for a in cluster])),
            "sphericity_avg": float(np.mean([a.sphericity for a in cluster])),
        })

    return mask_vol, cluster_stats


# =========================
# 主流程
# =========================
def preprocess_lidc():
    scans = pl.query(pl.Scan).all()
    print(f"Found {len(scans)} scans in database.")

    index_records = []

    for scan in tqdm(scans, desc="Preprocessing LIDC"):
        scan_id = scan.patient_id

        img_path  = os.path.join(IMAGE_DIR, f"{scan_id}_image.nii.gz")
        mask_path = os.path.join(MASK_DIR,  f"{scan_id}_mask.nii.gz")
        meta_path = os.path.join(META_DIR,  f"{scan_id}_meta.json")

        try:
            try:
                vol = scan.to_volume(verbose=False)
            except Exception as e:
                if "Couldn't find DICOM files" in str(e):
                    continue
                else:
                    raise
            affine = make_affine_from_spacing(get_spacing_xyz(scan))

            mask_vol, cluster_stats = build_mask_and_cluster_stats(scan, vol.shape)
            text_report = generate_text_report(cluster_stats)

            if not cfg.keep_negative_scans and len(cluster_stats) == 0:
                continue

            nib.save(nib.Nifti1Image(vol.astype(np.int16), affine), img_path)
            nib.save(nib.Nifti1Image(mask_vol.astype(np.uint8), affine), mask_path)

            meta = {
                "scan_id": scan_id,
                "image_path": img_path,
                "mask_path": mask_path,
                "text_report": text_report,
                "cluster_stats": cluster_stats,
            }

            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            index_records.append(meta)

        except Exception as e:
            print(f"\nFailed on {scan_id}: {e}")
            logging.exception(f"Failed on {scan_id}")

    with open(INDEX_JSON, "w") as f:
        json.dump(index_records, f, indent=2)

    print(f"\nPreprocess complete! Saved {len(index_records)} records.")


if __name__ == "__main__":
    preprocess_lidc()
