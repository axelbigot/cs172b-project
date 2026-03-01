import torch
import torch.nn as nn
import numpy as np
import librosa
from typing import List, Callable

from src.fma import VariableFMADataset
from src.common import AbstractFMAGenreModule

class MelCNNFMAModel(AbstractFMAGenreModule):
	@classmethod
	def train_generic(cls, train_dataset, val_dataset):
		model = cls(train_dataset.num_classes)
		model.fma_train(train_dataset, val_dataset, batch_size=8, num_epochs=1000)

	@classmethod
	def test_generic(cls, test_dataset):
		model = cls(test_dataset.num_classes)
		acc = model.fma_test(test_dataset)
		print(f'Test accuracy: {acc*100:.2f}%')

	@classmethod
	def name(cls):
		return 'mel-cnn'

	def __init__(self, num_classes: int, sr: int = 22050, n_mels: int = 128,
		n_fft: int = 2048, hop_length: int = 512, conv_channels: tuple[int,...]=(16,32,64), **kwargs):
		super().__init__(**kwargs)
		self.sr = sr
		self.n_mels = n_mels
		self.n_fft = n_fft
		self.hop_length = hop_length
		self.mel_cache_ = {}
		layers = []
		in_ch = 1
		for out_ch in conv_channels:
			layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False))
			layers.append(nn.BatchNorm2d(out_ch))
			layers.append(nn.ReLU(inplace=True))
			layers.append(nn.Dropout2d(0.2))
			layers.append(nn.MaxPool2d((2,1)))
			in_ch = out_ch
		layers.append(nn.AdaptiveAvgPool2d((1,None)))
		self.feature_extractor = nn.Sequential(*layers)
		self.dropout = nn.Dropout(0.5)
		self.classifier = nn.Linear(conv_channels[-1], num_classes)

	def _compute_mel(self, audio: np.ndarray) -> torch.Tensor:
		S = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, power=2.0)
		S_db = librosa.power_to_db(S, ref=np.max)
		S_norm = (S_db - np.mean(S_db)) / max(np.std(S_db),1e-6)
		return torch.from_numpy(S_norm.astype(np.float32)).unsqueeze(0)

	def transform_batch(self, dataset: VariableFMADataset, x: List[Callable[..., torch.Tensor]], ids: List[int]) -> torch.Tensor:
		split_cache = self.mel_cache_.setdefault(dataset.split,{})
		device = next(self.parameters()).device
		mels = []
		for aid, load_fn in zip(ids, x):
			if aid not in split_cache:
				audio = load_fn('cpu')
				audio = audio.numpy()
				split_cache[aid] = self._compute_mel(audio)
			mels.append(split_cache[aid])
		max_t = max(m.shape[2] for m in mels)
		padded = [torch.cat([m, torch.zeros((1,self.n_mels,max_t - m.shape[2]))], dim=2) if m.shape[2]<max_t else m for m in mels]
		return torch.stack([m.to(device) for m in padded], dim=0)

	def forward(self, batch_X: torch.Tensor, ids: List[int]) -> torch.Tensor:
		x = batch_X.to(next(self.parameters()).device)
		feat = self.feature_extractor(x)
		feat = feat.mean(dim=3).squeeze(2)
		feat = self.dropout(feat)
		return self.classifier(feat)
	