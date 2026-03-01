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
from src.fma.dataset_analyzer import DatasetAnalyzer
from src.constants import *


DATA_DIR = Path('data')

class VariableFMADataset(Dataset):
	metadata_cache: dict[str, pd.DataFrame] = {}

	def __init__(self,
		fma_metadata_zip: PathLike = 'fma_metadata.zip',
		fma_small_zip: PathLike = 'fma_small.zip',
		fma_metadata_out: PathLike = DATA_DIR,
		fma_small_out: PathLike = DATA_DIR,
		sampling_rate=22050,
		audio_min_sec=10,
		audio_max_sec=30,
		random_seed=RANDOM_SEED,
		genre_encoder=None,
		split='training', # 'training'|'validation'|'test'
		downsample_frac: float = 1
	):
		super().__init__()

		self.random_seed_ = random_seed
		self.sampling_rate_ = sampling_rate
		self.audio_min_sec_ = audio_min_sec
		self.audio_max_sec_ = audio_max_sec

		self.fma_metadata_out_ = Path(fma_metadata_out) / 'fma_metadata'
		self.fma_small_out_ = Path(fma_small_out) / 'fma_small'

		splits = ['training', 'validation', 'test']
		if split not in splits:
			raise RuntimeError(f'Unknown split {split}. Supported values are: {splits}')
		
		logging.info(f'[DATASET] Creating {split} dataset')

		self.idstr_ = f'split-{split}_seed-{RANDOM_SEED}_min-{self.audio_min_sec_}_max-{self.audio_max_sec_}_sr-{self.sampling_rate_}_frac-{downsample_frac}'

		cache_out_ = Path(DATA_DIR / f'VariableFMADataset-audio-cache-{self.idstr_}.pt')

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

			if path not in self.metadata_cache:
				self.metadata_cache[path] = fma_utils.load(path)
			
			return self.metadata_cache[path]
		
		pd_tracks = load_metata_csv('tracks.csv')
		# pd_genres = load_metata_csv('genres.csv')
		# pd_features = load_metata_csv('features.csv')
		# pd_echonest = load_metata_csv('echonest.csv')

		subset_mask = pd_tracks['set', 'subset'] <= 'small'
		# valid_tracks = pd_tracks.index.intersection(pd_echonest.index)
		index = pd_tracks.index[subset_mask]#.intersection(valid_tracks)

		pd_tracks = pd_tracks.loc[index]
		pd_tracks = pd_tracks[pd_tracks['set', 'split'] == split]

		if downsample_frac == 1.0:
			pass
		elif not (0 < downsample_frac < 1.0):
			raise ValueError(f'downsample_frac must be in [0, 1]')
		else:
			logging.info(f'[DATASET] Downsampling to {downsample_frac:.2f} fraction (stratified by genre)')

			rng = np.random.RandomState(self.random_seed_)
			
			sampled_indices = []
			for genre, group in pd_tracks.groupby(('track', 'genre_top')):
				n_samples = int(max(1, int(len(group)) * downsample_frac))

				sampled = group.sample(
					n=n_samples,
					random_state=rng
				)

				sampled_indices.append(sampled.index)
			
			new_index = pd.Index(np.concatenate(sampled_indices))
			pd_tracks = pd_tracks.loc[new_index]

		# pd_features = pd_features.loc[index]
		# pd_echonest = pd_echonest.loc[index]

		# np.testing.assert_array_equal(pd_features.index, pd_tracks.index)
		# assert pd_echonest.index.isin(pd_tracks.index).all()

		if split == 'training':
			self.genre_encoder = LabelEncoder()
			self.genre_encoder.fit(pd_tracks[('track', 'genre_top')])
		else:
			if genre_encoder is None:
				raise RuntimeError(f'For non train split, genre_encoder must be provided with the encoder from the train split')
			self.genre_encoder = genre_encoder

		pd_tracks['genre_encoded'] = self.genre_encoder.transform(pd_tracks[('track', 'genre_top')])
		self.track_genres = torch.tensor(pd_tracks['genre_encoded'].values, dtype=torch.long)

		self.loader_ = fma_utils.LibrosaLoader(sampling_rate=sampling_rate)

		self.track_ids_ = []
		self.audio_start_pos_ = []
		self.audio_end_pos_ = []

		logging.info(f'[DATASET] Preloading {len(pd_tracks)} audio files into memory')
		if cache_out_.exists():
			logging.info(f'[DATASET] Audio files retrieved from cache {cache_out_}')
			cache_dict = torch.load(cache_out_)

			self.audio_cache_ = cache_dict['audio_cache']
			self.track_ids_ = cache_dict['track_ids']
			self.audio_start_pos_ = cache_dict['start_pos']
			self.audio_end_pos_ = cache_dict['end_pos']
		else:
			self.audio_cache_: List[Tuple[torch.FloatTensor, torch.LongTensor]] = []
			for tid, genre in tqdm(zip(pd_tracks.index, self.track_genres), total=len(pd_tracks), desc='Preloading'):
				result = self.subsample_audio_(tid)
				if result is None:
					continue
				
				crop, st, end = result

				self.audio_cache_.append((crop, genre))
				self.track_ids_.append(tid)
				self.audio_start_pos_.append(st)
				self.audio_end_pos_.append(end)

			logging.info(f'Saving preloaded audio files to cache {cache_out_}')
			torch.save({
				"audio_cache": self.audio_cache_,
				"track_ids": self.track_ids_,
				"start_pos": self.audio_start_pos_,
				"end_pos": self.audio_end_pos_,
			}, cache_out_)

		self.track_genres = torch.tensor([g for _, g in self.audio_cache_], dtype=torch.long)

	def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.LongTensor, int]:
		audio, genre = self.audio_cache_[index]
		return audio, genre, index
	
	def __len__(self):
		return len(self.audio_cache_)
	
	def analyzer(self) -> DatasetAnalyzer:
		return DatasetAnalyzer(
			name=f'{self.__class__.__name__}',
			idstr=f'{self.__class__.__name__}_{self.idstr_}',
			sampling_rate=self.sampling_rate_,
			audio_tensors=[a for a, _ in self.audio_cache_],
			track_genres=self.track_genres,
			genre_encoder=self.genre_encoder,
			audio_min_sec=self.audio_min_sec_,
			audio_max_sec=self.audio_max_sec_,
			audio_start_pos=self.audio_start_pos_,
			audio_end_pos=self.audio_end_pos_,
		)

	def subsample_audio_(self, tid) -> Tuple[torch.FloatTensor, int, int] | None:
		seed = self.compute_seed_(tid)
		rng = np.random.RandomState(seed)

		path = fma_utils.get_audio_path(self.fma_small_out_, tid)
		try:
			audio_bytes = self.loader_.load(path)
			audio_tensor = torch.tensor(audio_bytes, dtype=torch.float32)
		except Exception as e:
			logging.warning(f'[DATASET] Skipping audio file {path} (tid: {tid}) due to error: {e}')
			return None

		seg_len = int(rng.uniform(self.audio_min_sec_, self.audio_max_sec_) * self.sampling_rate_)
		if len(audio_tensor) >= seg_len:
			st = rng.randint(0, len(audio_tensor) - seg_len + 1)
			end = st + seg_len
			crop = audio_tensor[st:end]
		else:
			logging.warning(
					f'Audio bytes too short ({len(audio_tensor)}) (tid: {tid}) '
					f'for requested segment length ({seg_len}). Using entire audio as fallback'
			)
			st = 0
			end = len(audio_tensor)
			crop = audio_tensor

		return crop, st, end

	def compute_seed_(self, tid: int) -> int:
		k = f'{self.random_seed_}_{tid}'.encode('utf8')
		digest = hashlib.sha1(k).digest()
		seed = int.from_bytes(digest[:4], 'big')
		return seed
