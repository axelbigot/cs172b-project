from typing import List, Tuple
import torch
import torch.nn as nn
import math

from src.fma import VariableFMADataset
from src.common import AbstractFMAGenreModule

"""
def mel_collate(batch: List[Tuple[torch.Tensor, torch.LongTensor, int]]):
	mels = torch.stack([x for x, _, _ in batch], dim=0)
	labels = torch.stack([y for _, y, _ in batch], dim=0)
	ids = [i for _, _, i in batch]

	padding_mask = (mels.sum(dim=2) == 0)

	return mels, labels, ids, padding_mask
"""


def mel_collate(batch: List[Tuple[torch.Tensor, torch.LongTensor, int]]):
	mels = torch.stack([x for x, _, _ in batch], dim=0)
	labels = torch.stack([y for _, y, _ in batch], dim=0)
	ids = [i for _, _, i in batch]
	
	return mels, labels, ids

class PositionalEncoding(nn.Module):
	def __init__(self, d_model: int, max_len: int = 5000):
		super().__init__()
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x + self.pe[:x.size(1), :]

class MelCNNTransformerFMAModel(AbstractFMAGenreModule):
	@classmethod
	def collate_fn(cls):
		return mel_collate
			
	@classmethod
	def train_generic(cls, train_dataset: VariableFMADataset, val_dataset: VariableFMADataset, **kwargs):
		model = cls(train_dataset.num_classes, **kwargs)

		optimizer = torch.optim.Adam(
			model.parameters(),
			lr=3e-4,
			weight_decay=1e-4
		)

		criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.1)

		model.fma_train(
			train_dataset,
			val_dataset,
			batch_size=32,
			num_epochs=100,
			optimizer=optimizer,
			criterion=criterion
		)

	@classmethod
	def test_generic(cls, test_dataset: VariableFMADataset, **kwargs):
		model = cls(test_dataset.num_classes, **kwargs)
		acc = model.fma_test(test_dataset)
		print(f'Test accuracy: {acc*100:.2f}%')

	@classmethod
	def name(cls):
		return 'mel-cnn-transformer'

	def __init__(self, num_classes: int, conv_channels=(32, 64, 128), n_heads=8, **kwargs):
		super().__init__(**kwargs)
		
		layers = []
		in_ch = 1
		for out_ch in conv_channels:
			layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False))
			layers.append(nn.BatchNorm2d(out_ch))
			layers.append(nn.ReLU(inplace=True))
			layers.append(nn.MaxPool2d(2))
			in_ch = out_ch
		
		self.feature_extractor = nn.Sequential(*layers)

		self.d_model = conv_channels[-1]
		self.pos_encoder = PositionalEncoding(self.d_model)

		encoder_layer = nn.TransformerEncoderLayer(
			d_model=self.d_model,
			nhead=n_heads,
			dim_feedforward=d_model*2,
			dropout=0.2,
			batch_first=True
		)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

		self.dropout = nn.Dropout(0.4)
		self.classifier = nn.Linear(self.d_model, num_classes)

	def forward(self, batch_X: torch.Tensor, ids: List[int] = None) -> torch.Tensor:
		device = next(self.parameters()).device
		x = batch_X.to(device)

		if x.ndim == 3:
			x = x.unsqueeze(1)

		feat = self.feature_extractor(x)
		feat = torch.mean(feat, dim=2)
		feat = feat.permute(0, 2, 1)
		feat = self.pos_encoder(feat)
		feat = self.transformer_encoder(feat)

		out = feat.mean(dim=1)
		out = self.dropout(out)

		return self.classifier(out)
