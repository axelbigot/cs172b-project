"""
Torch Dataset implementation for FMA.

Table descriptions: https://nbviewer.org/github/mdeff/fma/blob/outputs/usage.ipynb
"""
import pandas as pd
import numpy as np
import torch
import logging
import hashlib
import os

from torch.utils.data import Dataset
from os import PathLike
from pathlib import Path
from zipfile import ZipFile
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from typing import List, Tuple
from tqdm import tqdm

import src.fma.fma_utils as fma_utils
from src.constants import *


DATA_DIR = Path('data')

class VariableFMADataset(Dataset):
	def __init__(self,
		fma_metadata_zip: PathLike = 'fma_metadata.zip',
		fma_small_zip: PathLike = 'fma_small.zip',
		fma_metadata_out: PathLike = DATA_DIR,
		fma_small_out: PathLike = DATA_DIR,
		sampling_rate=22050,
		audio_min_sec=10,
		audio_max_sec=30,
		random_seed=RANDOM_SEED
	):
		super().__init__()

		self.random_seed_ = random_seed
		self.sampling_rate_ = sampling_rate
		self.audio_min_sec_ = audio_min_sec
		self.audio_max_sec_ = audio_max_sec

		self.fma_metadata_out_ = Path(fma_metadata_out) / 'fma_metadata'
		self.fma_small_out_ = Path(fma_small_out) / 'fma_small'

		cache_out_ = Path(DATA_DIR / f'VariableFMADataset-audio-cache.pt')

		fma_metadata_zip = Path(fma_metadata_zip)
		fma_small_zip = Path(fma_small_zip)

		def extract_zip(zip: Path, out: Path, check: Path):
			logging.info(f'[DATASET] Extracting from {zip} to {out}')
			if check.exists():
				logging.info(f'[DATASET] output dir {check} already exists. {zip} assumed extracted, skipping ...')
			else:
				out.mkdir(parents=True, exist_ok=True)

				with ZipFile(zip, 'r') as z:
					z.extractall(out)

		extract_zip(fma_metadata_zip, fma_metadata_out, self.fma_metadata_out_)
		extract_zip(fma_small_zip, fma_small_out, self.fma_small_out_)

		def load_metata_csv(csv_name: str) -> pd.DataFrame:
			path = self.fma_metadata_out_ / csv_name
			logging.info(f'[DATASET] Loading CSV {path} into memory')
			return fma_utils.load(path)
		
		pd_tracks = load_metata_csv('tracks.csv')
		# pd_genres = load_metata_csv('genres.csv')
		# pd_features = load_metata_csv('features.csv')
		pd_echonest = load_metata_csv('echonest.csv')

		subset_mask = pd_tracks['set', 'subset'] <= 'small'
		valid_tracks = pd_tracks.index.intersection(pd_echonest.index)
		index = pd_tracks.index[subset_mask].intersection(valid_tracks)

		pd_tracks = pd_tracks.loc[index]
		# pd_features = pd_features.loc[index]
		# pd_echonest = pd_echonest.loc[index]

		# np.testing.assert_array_equal(pd_features.index, pd_tracks.index)
		# assert pd_echonest.index.isin(pd_tracks.index).all()
		
		genre_encoder = LabelEncoder()
		genre_encoder.fit(pd_tracks[('track', 'genre_top')])
		pd_tracks['genre_encoded'] = genre_encoder.transform(pd_tracks[('track', 'genre_top')])

		self.track_genres_ = torch.tensor(pd_tracks['genre_encoded'].values, dtype=torch.long)
		self.loader_ = fma_utils.LibrosaLoader(sampling_rate=sampling_rate)

		logging.info(f'[DATASET] Preloading {len(pd_tracks)} audio files into memory')
		if cache_out_.exists():
			self.audio_cache_ = torch.load(cache_out_)
		else:
			self.audio_cache_: List[Tuple[torch.FloatTensor, torch.LongTensor]] = []
			for index in tqdm(range(len(pd_tracks)), desc='Preloading'):
				crop = self.subsample_audio_(pd_tracks, index)
				genre = self.track_genres_[index]
				self.audio_cache_.append((crop, genre))
			torch.save(self.audio_cache_, cache_out_)

	def __getitem__(self, index) -> Tuple[torch.FloatTensor, torch.LongTensor]:
		audio, genre = self.audio_cache_[index]
		return audio, genre
	
	def __len__(self):
		return len(self.audio_cache_)
	
	def subsample_audio_(self, pd_tracks, index):
		seed = self.compute_seed_(index)
		tid = pd_tracks.index[index]
		rng = np.random.RandomState(seed)

		path = fma_utils.get_audio_path(self.fma_small_out_, tid)
		try:
			audio_bytes = self.loader_.load(path)
		except Exception as e:
			logging.warning(f'[DATASET] Skipping audio file {path} due to error: {e}')
		audio_tensor = torch.tensor(audio_bytes, dtype=torch.float32)

		seg_len = int(rng.uniform(self.audio_min_sec_, self.audio_max_sec_) * self.sampling_rate_)
		if len(audio_tensor) >= seg_len:
			st = rng.randint(0, len(audio_tensor) - seg_len + 1)
			crop = audio_tensor[st:st + seg_len]
		else:
			logging.warning(f'Audio bytes too short ({len(audio_tensor)}) for requested segment length ({seg_len}). Using entire audio as a fallback')
			crop = audio_tensor
		return crop

	def compute_seed_(self, idx: int) -> int:
		k = f'{self.random_seed_}_{idx}'.encode('utf8')
		digest = hashlib.sha1(k).digest()
		seed = int.from_bytes(digest[:4], 'big')
		return seed
