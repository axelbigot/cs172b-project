from src.fma.fma_dataset import VariableFMADataset
from pathlib import Path
import torch
import numpy as np
import librosa
import random
from typing import Tuple
import logging
from tqdm import tqdm
from src.constants import *


class MelPrecomputeMixin:
	def __init__(
		self,
		n_fft=2048,
		hop_length=512,
		n_mels=128,
		power=2.0,
		*args, **kwargs
	):
		super().__init__(*args, **kwargs)

		self.n_fft = n_fft
		self.hop_length = hop_length
		self.n_mels = n_mels
		self.power = power

		cache_path = (
			Path(DATA_DIRECTORY)
			/ f"mel_cache_nfft-{n_fft}_hop-{hop_length}_nmels-{n_mels}_pwr-{power}_{self.idstr_}.pt"
		)

		if cache_path.exists():
			logging.info(f"[MelDataset] Loading precomputed mels from {cache_path}")
			self.mels_, self.labels_ = torch.load(cache_path)
		else:
			logging.info(f"[MelDataset] Precomputing mels for {len(self.index_)} tracks")
			self._precompute(cache_path)

	def _precompute(self, cache_path: Path):
		mels = []
		labels = []
		max_t = 0

		for idx in tqdm(range(len(self.index_))):
			audio_loader, label, _ = super().__getitem__(idx)
			audio = audio_loader("cpu").numpy()
			mel = self._compute_mel(audio)

			mels.append(mel)
			labels.append(label)
			max_t = max(max_t, mel.shape[2])

		device = "cpu"

		self.mels_ = torch.stack(
			[
				torch.cat(
					[m, torch.zeros((1, m.shape[1], max_t - m.shape[2]), device=device)],
					dim=2,
				)
				if m.shape[2] < max_t
				else m
				for m in mels
			],
			dim=0,
		)

		self.labels_ = torch.stack(labels, dim=0)

		torch.save((self.mels_, self.labels_), cache_path)
		logging.info(f"[MelDataset] Saved precomputed mels to {cache_path}")

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.LongTensor, int]:
		return self.mels_[index], self.labels_[index], index

	def __len__(self):
		return len(self.mels_)

	def _compute_mel(self, audio: np.ndarray) -> torch.Tensor:
		S = librosa.feature.melspectrogram(
			y=audio,
			sr=self.sampling_rate_,
			n_fft=self.n_fft,
			hop_length=self.hop_length,
			n_mels=self.n_mels,
			power=self.power,
		)
		S_db = librosa.power_to_db(S, ref=np.max)
		S_norm = (S_db - np.mean(S_db)) / max(np.std(S_db), 1e-6)
		return torch.from_numpy(S_norm.astype(np.float32)).unsqueeze(0)
	
class MelAugmentMixin:
	def __init__(self, 
		time_mask_width: int = 15, 
		freq_mask_width: int = 6, 
		n_time_masks: int = 1, 
		n_freq_masks: int = 1, 
		apply_prob: float = 0.35, 
		*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		self.specaug_time_mask_width = time_mask_width
		self.specaug_freq_mask_width = freq_mask_width
		self.specaug_n_time_masks = n_time_masks
		self.specaug_n_freq_masks = n_freq_masks
		self.specaug_apply_prob = apply_prob

		logging.info(f'[MEL DATASET AUGMENT] Using augmented mel dataset with masking')

	def __getitem__(self, index: int):
		mel, label, idx = super().__getitem__(index)
		if self.split == 'training':
			seed = RANDOM_SEED + index + self.epoch
			rng = np.random.RandomState(seed)
			if rng.rand() < self.specaug_apply_prob:
				mel = self._apply_specaugment(mel)
		return mel, label, idx

	def _apply_specaugment(self, mel: torch.Tensor):
		_, n_mels, T = mel.shape
		mel_aug = mel.clone()
		for _ in range(self.specaug_n_freq_masks):
			f = random.randint(0, self.specaug_freq_mask_width)
			f0 = random.randint(0, max(0, n_mels - f))
			mel_aug[:, f0:f0+f, :] = 0.0
		for _ in range(self.specaug_n_time_masks):
			t = random.randint(0, self.specaug_time_mask_width)
			t0 = random.randint(0, max(0, T - t))
			mel_aug[:, :, t0:t0+t] = 0.0
		return mel_aug
