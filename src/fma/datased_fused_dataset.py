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

class DatasedFusedDataset(VariableFMADataset):
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
		fma_samples = len(self.audio_cache_)
		count_per_genre = fma_samples / len(self.track_genres.unique())
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

		if cache_out_.exists():
			logging.info(f'[DATASET] Noise files retrieved from cache {cache_out_}')
			cache_dict = torch.load(cache_out_)

			self.audio_cache_.extend(cache_dict['audio_cache'])
			self.track_ids_.extend(cache_dict['track_ids'])
			self.audio_start_pos_.extend(cache_dict['start_pos'])
			self.audio_end_pos_.extend(cache_dict['end_pos'])
		else:
			segments = self.generate_segments_for_split_(files, target_count)

			for st, end, tid in segments:
				self.track_ids_.append(tid)
				self.audio_start_pos_.append(st)
				self.audio_end_pos_.append(end)

			logging.info(f'[DATASET] Saving noise to cache {cache_out_}')
			torch.save({
				'audio_cache': self.audio_cache_[fma_samples:],
				'track_ids': self.track_ids_[fma_samples:],
				'start_pos': self.audio_start_pos_[fma_samples:],
				'end_pos': self.audio_end_pos_[fma_samples:],
			}, cache_out_)
		
		self.track_genres = torch.tensor([g for _, g in self.audio_cache_], dtype=torch.long)

	def create_encoder(self, current_encoder):
		classes = list(current_encoder.classes_)
		if NON_MUSIC_CLASS not in classes:
			classes = sorted(classes + [NON_MUSIC_CLASS])
			current_encoder = genre_encoder = LabelEncoder()
			current_encoder = genre_encoder.fit(classes)

		return current_encoder

	def generate_segments_for_split_(self, files: List[Tuple[Path,float]], target_count: int) -> List[Tuple[int,int,int]]:
		segments: List[Tuple[int,int,int]] = []
		if target_count <= 0:
			return segments

		base_index = int(max(self.track_ids_) + 1)

		logging.info(f'[DATASET] Subsampling initial noise files (first pass)')
		for path, dur in tqdm(files, total=target_count, desc='Subsampling'):
			if len(segments) >= target_count:
				break
			fp = str(path)
			
			audio_tensor = self.load_audio_(fp)
			track_id = base_index + len(segments)
			crop, st, end = self.subsample_audio_(self.compute_seed_(track_id), audio_tensor)
			segments.append((st, end, track_id))
			self.audio_cache_.append((crop, torch.tensor(self.nonmusic_label, dtype=torch.long)))

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
					track_id = base_index + len(segments)
					crop, st, end = self.subsample_audio_(self.compute_seed_(track_id), audio_tensor)
					segments.append((st, end, track_id))
					self.audio_cache_.append((crop, torch.tensor(self.nonmusic_label, dtype=torch.long)))
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
