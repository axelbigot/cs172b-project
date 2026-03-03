from typing import List, Tuple
import torch
import torch.nn as nn
from src.fma import VariableFMADataset
from src.common import AbstractFMAGenreModule


def mel_collate(batch: List[Tuple[torch.Tensor, torch.LongTensor, int]]):
		mels = torch.stack([x for x, _, _ in batch], dim=0)  # shape: (B, 1, n_mels, T)
		labels = torch.stack([y for _, y, _ in batch], dim=0)
		ids = [i for _, _, i in batch]
		return mels, labels, ids

class MelCNNFMAModel(AbstractFMAGenreModule):
	@classmethod
	def collate_fn(cls):
		return mel_collate
		
	@classmethod
	def train_generic(cls, train_dataset: VariableFMADataset, val_dataset: VariableFMADataset, **kwargs):
			model = cls(train_dataset.num_classes, **kwargs)
			optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
			model.fma_train(train_dataset, val_dataset, batch_size=16, num_epochs=150, early_stopping_patience=10, optimizer=optimizer)

	@classmethod
	def test_generic(cls, test_dataset: VariableFMADataset, **kwargs):
			model = cls(test_dataset.num_classes, **kwargs)
			acc = model.fma_test(test_dataset)
			print(f'Test accuracy: {acc*100:.2f}%')

	@classmethod
	def name(cls):
			return 'mel-cnn'

	def __init__(self, num_classes: int, conv_channels=(32,64,128), **kwargs):
		super().__init__(**kwargs)
		layers = []
		in_ch = 1
		for out_ch in conv_channels:
			layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm2d(out_ch))
			layers.append(nn.ReLU(inplace=True))
			layers.append(nn.MaxPool2d((2,2)))
			in_ch = out_ch
		layers.append(nn.AdaptiveAvgPool2d((1,None)))
		self.feature_extractor = nn.Sequential(*layers)
		self.dropout = nn.Dropout(0.45)
		# self.classifier = nn.Linear(conv_channels[-1], num_classes)
		self.classifier = nn.Linear(conv_channels[-1] * 2, num_classes)

	def forward(self, batch_X: torch.Tensor, ids: List[int] = None) -> torch.Tensor:
		device = next(self.parameters()).device
		x = batch_X.to(device)
		feat = self.feature_extractor(x)
		# feat = feat.mean(dim=3).squeeze(2)
		feat = feat.squeeze(2)                 # (B, C, T')
		mean_pool = feat.mean(dim=2)           # (B, C)
		max_pool = feat.max(dim=2).values      # (B, C)
		feat = torch.cat([mean_pool, max_pool], dim=1)  # (B, 2C)
		feat = self.dropout(feat)
		return self.classifier(feat)
