"""
Torch Dataset implementation for FMA.

Table descriptions: https://nbviewer.org/github/mdeff/fma/blob/outputs/usage.ipynb
"""
#testing for corrupt audio
import pandas as pd
import numpy as np
import torch
import logging
import hashlib

from torch.utils.data import Dataset
from os import PathLike
from pathlib import Path
from zipfile import ZipFile
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass

import src.fma.fma_utils as fma_utils
from src.constants import *

DATA_DIR = Path('data')

@dataclass
class FMATrack:
    audio: np.ndarray
    length: float
    track_id: int
    genre: int
    features: np.ndarray
    echonest: np.ndarray

class VariableFMADataset(Dataset):
    def __init__(self,
        fma_metadata_zip = 'data/fma_metadata.zip',
        fma_small_zip: PathLike = 'data/fma_small.zip',
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

        fma_metadata_zip = Path(fma_metadata_zip)
        fma_small_zip = Path(fma_small_zip)

        def extract_zip(zip_path: Path, out_path: Path, check: Path, overwrite=False):
            print(f"Starting extraction of {zip_path} to {out_path}...")
            if check.exists() and not overwrite:
                print(f"Output directory {check} already exists. Skipping extraction.")
                return
            out_path.mkdir(parents=True, exist_ok=True)
            with ZipFile(zip_path, 'r') as z:
                print("Zip file opened. Extracting files...")
                z.extractall(out_path)
            print("Extraction complete!")

        extract_zip(fma_metadata_zip, fma_metadata_out, self.fma_metadata_out_)
        extract_zip(fma_small_zip, fma_small_out, self.fma_small_out_)

        def load_metata_csv(csv_name: str) -> pd.DataFrame:
            path = self.fma_metadata_out_ / csv_name
            logging.info(f'[DATASET] Loading CSV {path} into memory')
            return fma_utils.load(path)
        
        self.tracks_ = load_metata_csv('tracks.csv')
        self.genres_ = load_metata_csv('genres.csv')
        self.features_ = load_metata_csv('features.csv')
        self.echonest_ = load_metata_csv('echonest.csv')

        subset_mask = self.tracks_['set', 'subset'] <= 'small'
        valid_tracks = self.tracks_.index.intersection(self.echonest_.index)
        index = self.tracks_.index[subset_mask].intersection(valid_tracks)

        self.tracks_ = self.tracks_.loc[index]
        self.features_ = self.features_.loc[index]
        self.echonest_ = self.echonest_.loc[index]

        np.testing.assert_array_equal(self.features_.index, self.tracks_.index)
        assert self.echonest_.index.isin(self.tracks_.index).all()

        self.loader_ = fma_utils.LibrosaLoader(sampling_rate=sampling_rate)

        genre_encoder = LabelEncoder()
        genre_encoder.fit(self.tracks_[('track', 'genre_top')])
        self.tracks_['genre_encoded'] = genre_encoder.transform(self.tracks_[('track', 'genre_top')])

    def __getitem__(self, index):
        tid = self.tracks_.index[index]
        path = fma_utils.get_audio_path(self.fma_small_out_, tid)

        # ----------------------------
        # Safe audio loading
        # ----------------------------
        try:
            audio_bytes = self.loader_.load(path)
        except Exception as e:
            logging.warning(f"[DATASET] Could not load track {tid} ({path}): {e}")
            # Replace corrupted audio with 1 second of silence
            audio_bytes = np.zeros(self.sampling_rate_, dtype=np.float32)

        track = self.tracks_.loc[[tid]]

        seed = self.compute_seed_(index)
        rng = np.random.RandomState(seed)

        crop = self.sample_segment_(audio_bytes, rng)

        metadata = FMATrack(
            audio=crop.astype(np.float32),
            length=float(len(crop) / self.sampling_rate_),
            track_id=int(tid),
            genre=int(self.tracks_.loc[tid, 'genre_encoded']),
            features=self.features_.loc[tid].values.astype(np.float32),
            echonest=pd.to_numeric(self.echonest_.loc[tid], errors='coerce').fillna(0).values.astype(np.float32)
        )

        return metadata
    
    def __len__(self):
        return len(self.tracks_)

    def compute_seed_(self, idx: int) -> int:
        k = f'{self.random_seed_}_{idx}'.encode('utf8')
        digest = hashlib.sha1(k).digest()
        seed = int.from_bytes(digest[:4], 'big')
        return seed
    
    def sample_segment_(self, audio_bytes: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        seg_len = int(rng.uniform(self.audio_min_sec_, self.audio_max_sec_) * self.sampling_rate_)

        if len(audio_bytes) >= seg_len:
            st = rng.randint(0, len(audio_bytes) - seg_len + 1)
            seg = audio_bytes[st:st + seg_len]
        else:
            logging.warning(f'Audio bytes too short for requested segment length ({seg_len})')
            seg = audio_bytes
        
        return seg
    

# import pandas as pd
# import numpy as np
# import torch
# import logging
# import hashlib

# from torch.utils.data import Dataset
# from os import PathLike
# from pathlib import Path
# from zipfile import ZipFile
# from sklearn.preprocessing import LabelEncoder
# from dataclasses import dataclass

# import src.fma.fma_utils as fma_utils
# from src.constants import *


# DATA_DIR = Path('data')

# @dataclass
# class FMATrack:
# 	audio: np.ndarray
# 	length: float
# 	track_id: int
# 	genre: int
# 	features: np.ndarray
# 	echonest: np.ndarray

# class VariableFMADataset(Dataset):
# 	def __init__(self,
# 		fma_metadata_zip = 'data/fma_metadata.zip',
# 		fma_small_zip: PathLike = 'data/fma_small.zip',
# 		fma_metadata_out: PathLike = DATA_DIR,
# 		fma_small_out: PathLike = DATA_DIR,
# 		sampling_rate=22050,
# 		audio_min_sec=10,
# 		audio_max_sec=30,
# 		random_seed=RANDOM_SEED
# 	):
# 		super().__init__()

# 		self.random_seed_ = random_seed
# 		self.sampling_rate_ = sampling_rate
# 		self.audio_min_sec_ = audio_min_sec
# 		self.audio_max_sec_ = audio_max_sec

# 		self.fma_metadata_out_ = Path(fma_metadata_out) / 'fma_metadata'
# 		self.fma_small_out_ = Path(fma_small_out) / 'fma_small'

# 		fma_metadata_zip = Path(fma_metadata_zip)
# 		fma_small_zip = Path(fma_small_zip)

# 		#made it visualize this step
# 		def extract_zip(zip_path: Path, out_path: Path, check: Path, overwrite=False):
# 			print(f"Starting extraction of {zip_path} to {out_path}...")

# 			if check.exists() and not overwrite:
# 				print(f"Output directory {check} already exists. Skipping extraction.")
# 				return

# 			out_path.mkdir(parents=True, exist_ok=True)

# 			with ZipFile(zip_path, 'r') as z:
# 				print("Zip file opened. Extracting files...")
# 				z.extractall(out_path)

# 			print("Extraction complete!")

					

# 		extract_zip(fma_metadata_zip, fma_metadata_out, self.fma_metadata_out_)
# 		extract_zip(fma_small_zip, fma_small_out, self.fma_small_out_)

# 		def load_metata_csv(csv_name: str) -> pd.DataFrame:
# 			path = self.fma_metadata_out_ / csv_name
# 			logging.info(f'[DATASET] Loading CSV {path} into memory')
# 			return fma_utils.load(path)
		
# 		self.tracks_ = load_metata_csv('tracks.csv')
# 		self.genres_ = load_metata_csv('genres.csv')
# 		self.features_ = load_metata_csv('features.csv')
# 		self.echonest_ = load_metata_csv('echonest.csv')

# 		subset_mask = self.tracks_['set', 'subset'] <= 'small'
# 		valid_tracks = self.tracks_.index.intersection(self.echonest_.index)
# 		index = self.tracks_.index[subset_mask].intersection(valid_tracks)

# 		self.tracks_ = self.tracks_.loc[index]
# 		self.features_ = self.features_.loc[index]
# 		self.echonest_ = self.echonest_.loc[index]

# 		np.testing.assert_array_equal(self.features_.index, self.tracks_.index)
# 		assert self.echonest_.index.isin(self.tracks_.index).all()

# 		self.loader_ = fma_utils.LibrosaLoader(sampling_rate=sampling_rate)
		
# 		genre_encoder = LabelEncoder()
# 		genre_encoder.fit(self.tracks_[('track', 'genre_top')])
# 		self.tracks_['genre_encoded'] = genre_encoder.transform(self.tracks_[('track', 'genre_top')])

# 	def __getitem__(self, index):
# 		tid = self.tracks_.index[index]
# 		path = fma_utils.get_audio_path(self.fma_small_out_, tid)

# 		audio_bytes = self.loader_.load(path)

# 		track = self.tracks_.loc[[tid]]

# 		seed = self.compute_seed_(index)
# 		rng = np.random.RandomState(seed)

# 		crop = self.sample_segment_(audio_bytes, rng)

# 		metadata = FMATrack(
# 			audio=crop.astype(np.float32),
# 			length=float(len(crop) / self.sampling_rate_),
# 			track_id=int(tid),
# 			genre=int(self.tracks_.loc[tid, 'genre_encoded']),
# 			features=self.features_.loc[tid].values.astype(np.float32),
# 			echonest=pd.to_numeric(self.echonest_.loc[tid], errors='coerce').fillna(0).values.astype(np.float32)
#     )

# 		return metadata
	
# 	def __len__(self):
# 		return len(self.tracks_)

# 	def compute_seed_(self, idx: int) -> int:
# 		k = f'{self.random_seed_}_{idx}'.encode('utf8')
# 		digest = hashlib.sha1(k).digest()
# 		seed = int.from_bytes(digest[:4], 'big')
# 		return seed
	
# 	def sample_segment_(self, audio_bytes: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
# 		seg_len = int(rng.uniform(self.audio_min_sec_, self.audio_max_sec_) * self.sampling_rate_)

# 		if len(audio_bytes) >= seg_len:
# 			st = rng.randint(0, len(audio_bytes) - seg_len + 1)
# 			seg = audio_bytes[st:st + seg_len]
# 		else:
# 			logging.warning(f'Audio bytes too short for requested segment length ({seg_len})')
# 			seg = audio_bytes
		
# 		return seg
