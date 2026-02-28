# src/models/mel_cnn_fma.py
import torch
import torch.nn as nn
import numpy as np
import librosa

from src.constants import *
from src.fma import *
from src.common import *

class MelCNNFMAModel(AbstractFMAGenreModule):
	@classmethod
	def train_generic(cls, train_dataset, val_dataset):
		model = cls()
		model.fma_train(train_dataset, val_dataset, batch_size=8, num_epochs=5000)

	@classmethod
	def name(cls):
		return 'mel-cnn'

	def __init__(
		self,
		sr: int = 22050,
		n_mels: int = 128,
		n_fft: int = 2048,
		hop_length: int = 512,
		conv_channels: tuple[int, ...] = (16, 32, 64),
		**kwargs
	):
		super().__init__(**kwargs)
		self.sr = sr
		self.n_mels = n_mels
		self.n_fft = n_fft
		self.hop_length = hop_length

		channels = conv_channels
		layers = []
		in_ch = 1
		for out_ch in channels:
			layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False))
			layers.append(nn.BatchNorm2d(out_ch))
			layers.append(nn.ReLU(inplace=True))
			layers.append(nn.MaxPool2d(kernel_size=(2, 2)))
			in_ch = out_ch

		layers.append(nn.AdaptiveAvgPool2d((1, 1)))

		self.feature_extractor = nn.Sequential(*layers)
		self.classifier = nn.Linear(channels[-1], NUM_CLASSES)

	def compute_mel(self, audio: np.ndarray) -> np.ndarray:
		if audio.dtype != np.float32:
			audio = audio.astype(np.float32)
		S = librosa.feature.melspectrogram(
			y=audio,
			sr=self.sr,
			n_fft=self.n_fft,
			hop_length=self.hop_length,
			n_mels=self.n_mels,
			power=2.0
		)
		S_db = librosa.power_to_db(S, ref=np.max)
		mean = float(np.mean(S_db))
		std = float(np.std(S_db))
		if std < 1e-6:
			std = 1.0
		S_norm = (S_db - mean) / std
		return S_norm

	def forward(self, batch_X: List[torch.FloatTensor]) -> torch.Tensor:
		mel_list = []
		for a in batch.audios:
			arr = a.detach().cpu().numpy().astype(np.float32)
			if arr.ndim > 1:
				arr = np.mean(arr, axis=0)
			mel = self.compute_mel(arr)
			tensor = torch.from_numpy(mel).unsqueeze(0)
			mel_list.append(tensor)

		max_t = max(m.shape[2] for m in mel_list)
		padded = []
		for m in mel_list:
			t = m.shape[2]
			if t < max_t:
				padding = torch.zeros((1, self.n_mels, max_t - t), dtype=m.dtype)
				m = torch.cat([m, padding], dim=2)
			padded.append(m)

		device = next(self.parameters()).device
		x = torch.stack(padded, dim=0).to(device)

		feat = self.feature_extractor(x)
		feat = feat.view(feat.size(0), -1)
		logits = self.classifier(feat)
		return logits
