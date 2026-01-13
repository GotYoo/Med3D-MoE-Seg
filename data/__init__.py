"""
Data module for Med3D-MoE-Seg.
Keep imports lightweight to avoid failing when optional datasets are missing.
"""

try:
	from .builder import build_dataloader  # noqa: F401
	__all__ = ['build_dataloader']
except Exception:
	# Soft-fail so modules that only need alignment dataset can still import.
	__all__ = []
