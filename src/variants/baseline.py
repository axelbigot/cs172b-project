from typing import List
import torch
import torch.nn as nn
from src.fma import VariableFMADataset
from src.common import AbstractFMAGenreModule


def mel_baseline_collate(batch: List[tuple]):
	x = torch.stack([x for x, _, _ in batch], dim=0)
	labels = torch.stack([y for _, y, _ in batch], dim=0)
	ids = [i for _, _, i in batch]
	return x, labels, ids

class MelMLPFMAModel(AbstractFMAGenreModule):
	@classmethod
	def collate_fn(cls):
		return mel_baseline_collate

	@classmethod
	def train_generic(cls, train_dataset: VariableFMADataset, val_dataset: VariableFMADataset, tag: str):
		model = cls(train_dataset.num_classes, tag=tag)
		model.fma_train(train_dataset, val_dataset, batch_size=32, num_epochs=200)

	@classmethod
	def test_generic(cls, test_dataset: VariableFMADataset):
		model = cls(test_dataset.num_classes)
		acc = model.fma_test(test_dataset)
		print(f'Test accuracy: {acc*100:.2f}%')

	@classmethod
	def name(cls):
		return 'mel-mlp'

	def __init__(self, num_classes: int, hidden_dim: int = 64, **kwargs):
		super().__init__(**kwargs)
		self.hidden_dim = hidden_dim
		self.fc1 = nn.Linear(128, hidden_dim)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.3)
		self.fc2 = nn.Linear(hidden_dim, num_classes)

	def forward(self, batch_X: torch.Tensor, ids: List[int] = None) -> torch.Tensor:
		device = next(self.parameters()).device
		x = batch_X.to(device)
		x = x.mean(dim=3).squeeze(1)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout(x)
		return self.fc2(x)
	