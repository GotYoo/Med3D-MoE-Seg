"""
Placeholder LIDC dataset to satisfy imports.
Full implementation was absent; this stub prevents import errors when Stage 1 uses alignment dataset.
If you need real LIDC loading, replace with a proper dataset.
"""

from typing import Dict, List, Tuple, Union
import torch
from .base_dataset import BaseMedicalDataset, DatasetRegistry


@DatasetRegistry.register('lidc')
class LIDCDataset(BaseMedicalDataset):
	def __init__(self, *args, **kwargs):
		super().__init__(data_source=kwargs.get('data_source', ''),
						 image_size=kwargs.get('image_size', (128, 128, 128)),
						 normalize=kwargs.get('normalize', True),
						 augmentation=kwargs.get('augmentation', False))
		# This is a stub; real data loading should be implemented if used.
		self._data = []

	def __len__(self) -> int:
		return len(self._data)

	def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, Dict]]:
		raise NotImplementedError("LIDCDataset stub: implement data loading before use.")

	def get_dataset_info(self) -> Dict[str, any]:
		return {
			'name': 'LIDC (stub)',
			'num_samples': len(self._data),
			'modality': 'CT',
			'task': 'segmentation',
			'classes': []
		}

	@staticmethod
	def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
		raise NotImplementedError("LIDCDataset stub: implement collate_fn before use.")
