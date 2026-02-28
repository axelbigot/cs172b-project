from src.fma.fma_dataset import VariableFMADataset
from src.constants import *
from typing import List
from tqdm import tqdm

import torch
import numpy as np
import logging


# Adds white noise samples
class NoiseVariableFMADataset(VariableFMADataset):
	def __init__(self, frac_noise_samples=0.2, *args, **kwargs):
		super().__init__(*args, **kwargs)

		cache_loc = Path(DATA_DIRECTORY / 'NoiseVariableFMADataset-audio-cache.pt')

		self.num_noise_samples_ = int(frac_noise_samples * len(self.audio_cache_))
		self.noise_label_ = int(self.track_genres_.max().item()) + 1

		logging.info(f'[DATASET] Generating {self.num_noise_samples_} white noise samples and adding them to the audio dataset')
		if cache_loc.exists():
			self.audio_cache_ = torch.load(cache_loc)
		else:
			self.noise_cache_ = self.generate_noise_samples_()

			for idx, noise in tqdm(enumerate(self.noise_cache_), desc='Subsampling Noise', total=len(self.noise_cache_)):
				self.audio_cache_.append(
					(self.subsample_noise_(noise, idx), torch.tensor(self.noise_label_, dtype=torch.long))
				)

			logging.info(f'Saving noise audio cache to {cache_loc}')
			torch.save(self.audio_cache_, cache_loc)

		logging.info(f'[DATASET] Proceeding with {len(self.audio_cache_)} total audio samples')

		self.track_genres_ = torch.cat([
			self.track_genres_,
			torch.full((self.num_noise_samples_,), self.noise_label_, dtype=torch.long)
		])

	def generate_noise_samples_(self) -> List[torch.FloatTensor]:
		max_len = int(self.audio_max_sec_ * self.sampling_rate_)
		rng = np.random.RandomState(RANDOM_SEED)
		noise_samples = []
		for _ in tqdm(range(self.num_noise_samples_), desc='Generating Noise'):
			noise = torch.from_numpy(rng.randn(max_len).astype(np.float32))
			noise_samples.append(noise)
		return noise_samples
	
	def subsample_noise_(self, noise: torch.FloatTensor, idx: int) -> torch.FloatTensor:
		seed = RANDOM_SEED + idx
		rng = np.random.RandomState(seed)

		min_len = int(self.audio_min_sec_ * self.sampling_rate_)
		max_len = min(int(self.audio_max_sec_ * self.sampling_rate_), len(noise))
		seg_len = rng.randint(min_len, max_len + 1)

		if len(noise) <= seg_len:
			return noise
		
		st = rng.randint(0, len(noise) - seg_len + 1)
		return noise[st:st + seg_len]
	