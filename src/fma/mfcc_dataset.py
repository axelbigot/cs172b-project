from src.fma.fma_dataset import VariableFMADataset
from pathlib import Path
import torch
import numpy as np
import librosa
from typing import Tuple
import logging
from tqdm import tqdm
from src.constants import *


class MfccPrecomputeMixin:
	def __init__(self, *args, mfcc_coeffs: int = 40, fft_window: int = 2048, hop_length: int = 512, **kwargs):
		super().__init__(*args, **kwargs)

		self.mfcc_coeffs = mfcc_coeffs
		self.fft_window = fft_window
		self.hop_length = hop_length

		cache_path = Path(DATA_DIRECTORY) / f"mfcc_cache_coeff-{mfcc_coeffs}_fft-{fft_window}_hop-{hop_length}_{self.idstr_}.pt"

		if cache_path.exists():
			logging.info(f"[MfccDataset] Loading precomputed mfccs from {cache_path}")
			self.mfccs_, self.labels_ = torch.load(cache_path)
		else:
			logging.info(f"[MfccDataset] Precomputing mfccs for {len(self.index_)} tracks")

			mfccs = []
			labels = []
			max_t = 0

			for idx in tqdm(range(len(self.index_))):
				audio_loader, label, _ = super().__getitem__(idx)
				audio = audio_loader("cpu").numpy()

				if audio.ndim > 1:
					audio = audio.mean(axis=0)

				mfcc = self._compute_mfcc(audio)
				mfccs.append(mfcc)
				labels.append(label)

				max_t = max(max_t, mfcc.shape[2])

			self.mfccs_ = torch.stack([
				torch.cat(
					[m, torch.zeros((1, self.mfcc_coeffs, max_t - m.shape[2]))],
					dim=2
				) if m.shape[2] < max_t else m
				for m in mfccs
			], dim=0)

			self.labels_ = torch.stack(labels, dim=0)

			torch.save((self.mfccs_, self.labels_), cache_path)
			logging.info(f"[MfccDataset] Saved precomputed mfccs to {cache_path}")

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.LongTensor, int]:
		return self.mfccs_[index], self.labels_[index], index

	def __len__(self):
		return len(self.mfccs_)

	def _compute_mfcc(self, audio: np.ndarray) -> torch.Tensor:
		mfcc_feat = librosa.feature.mfcc(
			y=audio,
			sr=self.sampling_rate_,
			n_mfcc=self.mfcc_coeffs,
			n_fft=self.fft_window,
			hop_length=self.hop_length
		)

		mean, std = np.mean(mfcc_feat), np.std(mfcc_feat)
		std = std if std > 1e-6 else 1.0
		mfcc_norm = (mfcc_feat - mean) / std

		return torch.from_numpy(mfcc_norm.astype(np.float32)).unsqueeze(0)
