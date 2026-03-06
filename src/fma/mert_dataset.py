"""
src/fma/mert_dataset.py

Loads raw waveforms from FMA at 24kHz for MERT.
Caches individual track files to disk on first run.

RAM preload strategy (auto-selected by dataset size):
  - FMA small  (~6,400 tracks)  : preload all into RAM for fast iteration
  - FMA medium (~25,000 tracks) : load from disk per batch (72GB RAM would OOM)
"""
from src.fma.fma_dataset import VariableFMADataset
from pathlib import Path
import torch
import numpy as np
import librosa
import logging
import gc
from typing import Tuple
from tqdm import tqdm
from src.constants import DATA_DIRECTORY

MERT_SAMPLE_RATE = 24000
MAX_DURATION_SEC = 30       # full 30 seconds
RAM_PRELOAD_THRESHOLD = 8000  # CHANGED: only preload to RAM if dataset <= this size


def load_audio_mert(audio: np.ndarray, orig_sr: int) -> torch.Tensor:
    if orig_sr != MERT_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=MERT_SAMPLE_RATE)
    max_samples = MAX_DURATION_SEC * MERT_SAMPLE_RATE
    audio = audio[:max_samples]
    return torch.from_numpy(audio.astype(np.float32))


class MERTDataset(VariableFMADataset):
    """
    Extends VariableFMADataset — precomputes 24kHz waveforms and caches
    each track as a separate file on disk.

    For small datasets (<=8000 tracks): preloads all into RAM for fast iteration.
    For large datasets (>8000 tracks): loads from disk per batch to avoid OOM.

    __getitem__ returns:
        waveform : (T,)  float32 tensor
        label    : ()    LongTensor
        index    : int
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        orig_sr = self.sampling_rate_
        self.cache_dir = Path(DATA_DIRECTORY) / f"mert_cache_{self.idstr_}"
        self.labels_path = self.cache_dir / "labels.pt"

        if self.labels_path.exists():
            logging.info(f"[MERTDataset] Found cache at {self.cache_dir}")
            self.wave_labels_ = torch.load(self.labels_path)
            self.valid_indices_ = list(range(len(self.wave_labels_)))
        else:
            logging.info(f"[MERTDataset] Precomputing waveforms for {len(self.index_)} tracks")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._precompute(orig_sr)

        # CHANGED: only preload to RAM for small datasets to avoid OOM on medium/large
        self.use_ram_cache_ = len(self.wave_labels_) <= RAM_PRELOAD_THRESHOLD
        if self.use_ram_cache_:
            logging.info(f"[MERTDataset] Preloading {len(self.wave_labels_)} waveforms into RAM...")
            self.ram_cache_ = [
                torch.load(self.cache_dir / f"track_{self.valid_indices_[i]}.pt")
                for i in tqdm(range(len(self.wave_labels_)), desc="Loading to RAM")
            ]
            logging.info("[MERTDataset] RAM preload complete")
        else:
            # CHANGED: skip RAM preload for large datasets, load from disk per batch
            logging.info(f"[MERTDataset] Large dataset ({len(self.wave_labels_)} tracks) — loading from disk per batch")
            self.ram_cache_ = None

    def _precompute(self, orig_sr: int):
        labels_list = []
        valid_indices = []

        for idx in tqdm(range(len(self.index_)), desc="Precomputing MERT waveforms"):
            try:
                audio_loader, label, _ = super().__getitem__(idx)
                audio = audio_loader("cpu").numpy()
                waveform = load_audio_mert(audio, orig_sr)
                torch.save(waveform, self.cache_dir / f"track_{idx}.pt")
                labels_list.append(label)
                valid_indices.append(idx)
                del waveform, audio
                gc.collect()
            except Exception as e:
                logging.warning(f"[MERTDataset] Skipping track {idx}: {e}")

        self.wave_labels_ = torch.stack(labels_list)
        self.valid_indices_ = valid_indices
        torch.save(self.wave_labels_, self.labels_path)
        logging.info(f"[MERTDataset] Cached {len(valid_indices)} tracks to {self.cache_dir}")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.use_ram_cache_:
            # CHANGED: use RAM cache for small datasets
            return self.ram_cache_[index], self.wave_labels_[index], index
        else:
            # CHANGED: load from disk for large datasets to avoid OOM
            track_idx = self.valid_indices_[index]
            waveform = torch.load(self.cache_dir / f"track_{track_idx}.pt")
            return waveform, self.wave_labels_[index], index

    def __len__(self):
        return len(self.wave_labels_)


def collate_mert(batch):
    """
    Pad waveforms to same length within batch.

    Returns:
        waveforms      : (B, T)   padded float32
        attention_mask : (B, T)   1 where real, 0 where padded
        labels         : (B,)     LongTensor
        ids            : list[int]
    """
    waveforms, labels, ids = zip(*batch)
    max_len = max(w.shape[0] for w in waveforms)

    padded = torch.zeros(len(waveforms), max_len)
    mask   = torch.zeros(len(waveforms), max_len)

    for i, w in enumerate(waveforms):
        padded[i, :w.shape[0]] = w
        mask[i, :w.shape[0]]   = 1.0

    return (
        padded,
        mask,
        torch.stack(labels).long(),
        list(ids),
    )