from src.fma.fma_dataset import VariableFMADataset
from src.constants import *
from typing import List
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from os import PathLike
from typing import Tuple
from pathlib import Path

import torch
import numpy as np
import logging
import librosa
import soundfile as sf


NON_MUSIC_CLASS = 'X-Non-Music'

class DatasedFusedMixin:
	def __init__(self,
		datased_zip: PathLike = 'DataSED - Dataset for Sound Event Detection of environmental noise.zip',
		datased_out = DATA_DIRECTORY,
		noise_count_factor: float = 1.0,
		*args, **kwargs
	):
		super().__init__(*args, **kwargs)

		logging.info(f'[DATASET] Building DatasedFusedDataset {self.split} split')

		self.idstr_ = f'{self.idstr_}_noise-{noise_count_factor}'
		cache_out_ = Path(DATA_DIRECTORY / f'DatasedFusedDataset-audio-cache-{self.idstr_}.pt') 

		datased_extracted_dir = DATA_DIRECTORY / 'DataSED - DataSED - Dataset for Sound Event Detection of environmental noise'
		self.extract_zip_(datased_zip, datased_out, datased_extracted_dir)

		# where the wav files will be
		datased_wav_dir = datased_extracted_dir / 'SED_wav'
		files = self.get_datased_files_(datased_wav_dir)

		train_files, val_files, test_files = self.assign_dataset_split_(files)
		self.fma_samples = len(self.index_)
		count_per_genre = self.fma_samples / (self.num_classes - 1)
		target_count = int(count_per_genre * noise_count_factor)

		if self.split == 'training':
			files = train_files
		elif self.split == 'validation':
			files = val_files
		else:
			files = test_files

		logging.info(f'[DATASET] Reserved {len(files)} for {self.split} split')

		if NON_MUSIC_CLASS not in self.genre_encoder.classes_:
			raise RuntimeError(f'Genre encoder provided is invalid and does not contain {NON_MUSIC_CLASS}')
		
		self.nonmusic_label = int(self.genre_encoder.transform([NON_MUSIC_CLASS])[0])

		self.noise_index_: List[dict] = []
		if cache_out_.exists():
			logging.info(f'[DATASET] Noise files retrieved from cache {cache_out_}')
			self.noise_index_ = torch.load(cache_out_)
		else:
			segments = self.generate_segments_for_split_(files, target_count)

			for path, st, end, tid in segments:
				self.noise_index_.append({
					'track_id': tid,
					'path': path,
					'start': st,
					'end': end,
					'label': int(self.nonmusic_label)
				})

			logging.info(f'[DATASET] Saving noise to cache {cache_out_}')
			torch.save(self.noise_index_, cache_out_)
		
		self.index_.extend(self.noise_index_)

	def create_encoder(self, current_encoder):
		classes = list(current_encoder.classes_)
		if NON_MUSIC_CLASS not in classes:
			classes = sorted(classes + [NON_MUSIC_CLASS])
			current_encoder = LabelEncoder()
			current_encoder = current_encoder.fit(classes)

		return current_encoder

	def generate_segments_for_split_(self, files: List[Tuple[Path,float]], target_count: int) -> List[Tuple[int,int,int]]:
		segments: List[Tuple[str, int, int, int]] = []
		if target_count <= 0:
			return segments

		logging.info(f'[DATASET] Subsampling initial noise files (first pass)')
		for path, dur in tqdm(files, total=target_count, desc='Subsampling'):
			if len(segments) >= target_count:
				break
			fp = str(path)
			
			audio_tensor = self.load_audio_(fp)
			track_id = self.fma_samples + len(segments)
			_, st, end = self.subsample_audio_(self.compute_seed_(track_id), audio_tensor)
			segments.append((str(path), st, end, track_id))

		min_split_len = max(40, int(self.audio_max_sec_ * 1.5))
		candidates = [f for f in files if f[1] >= min_split_len]
		candidates_sorted_desc = sorted(candidates, key=lambda x: (-x[1], str(x[0])))

		if len(segments) >= target_count:
			return segments

		pass_num = 1
		logging.info(f'[DATASET] Splitting larger noise files until quota ({target_count}) is met')
		with tqdm(total=target_count, desc='Splitting') as pbar:
			while len(segments) < target_count:
				made_progress = False
				for path, dur in candidates_sorted_desc:
					if len(segments) >= target_count:
						break
					fp = str(path)

					audio_tensor = self.load_audio_(fp)
					track_id = self.fma_samples + len(segments)
					_, st, end = self.subsample_audio_(self.compute_seed_(track_id), audio_tensor)
					segments.append((str(path), st, end, track_id))
					made_progress = True
				if made_progress:
					pbar.n = len(segments)
					pbar.refresh()
				else:
					break
				pass_num += 1

		if len(segments) < target_count:
			logging.warning(f'[DATASET] Requested {target_count} segments but produced {len(segments)}')
		return segments

	def get_datased_files_(self, root: Path) -> List[Tuple[Path, float]]:
		files = []
		for p in sorted(root.rglob('*wav')):
			try:
				info = sf.info(str(p))
				dur = float(info.frames) / float(info.samplerate)
			except Exception as e:
				logging.warning(f'[DATASET] skipping {p} due to error reading metadata: {e}')
				continue

			if dur >= self.audio_min_sec_:
				files.append((p, dur))

		return files

	def assign_dataset_split_(self, files):
		files = sorted(files, key=lambda x: x[1])
		train, val, test = [], [], []

		for i, f in enumerate(files):
			mod = i % 10
			if mod < 8:
				train.append(f)
			elif mod == 8:
				val.append(f)
			else:
				test.append(f)

		return train, val, test

	def load_audio_(self, filepath: str) -> torch.FloatTensor:
		audio, _ = librosa.load(filepath, sr=self.sampling_rate_, mono=True)
		return torch.tensor(audio, dtype=torch.float32)
