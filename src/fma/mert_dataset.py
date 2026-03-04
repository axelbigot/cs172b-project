"""
src/fma/mert_dataset.py

Loads raw waveforms from FMA small at 24kHz for MERT.
Caches to disk on first run — subsequent runs load instantly.
"""
from src.fma.fma_dataset import VariableFMADataset
from pathlib import Path
import torch
import numpy as np
import librosa
import logging
from typing import Tuple, List
from tqdm import tqdm
from src.constants import DATA_DIRECTORY

MERT_SAMPLE_RATE = 24000   # MERT-v1 requires 24kHz
MAX_DURATION_SEC = 30      # cap audio length


def load_audio_mert(audio: np.ndarray, orig_sr: int) -> torch.Tensor:
    """
    Resample audio to 24kHz and return as float32 tensor (T,).
    Clips to MAX_DURATION_SEC to keep memory bounded.
    """
    if orig_sr != MERT_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=MERT_SAMPLE_RATE)
    max_samples = MAX_DURATION_SEC * MERT_SAMPLE_RATE
    audio = audio[:max_samples]
    return torch.from_numpy(audio.astype(np.float32))


class MERTDataset(VariableFMADataset):
    """
    Extends VariableFMADataset — precomputes 24kHz waveforms and caches to disk.

    __getitem__ returns:
        waveform : (T,)  float32 tensor
        label    : ()    LongTensor
        index    : int
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        orig_sr = self.sampling_rate_
        cache_path = Path(DATA_DIRECTORY) / f"mert_waveform_cache_{self.idstr_}.pt"

        if cache_path.exists():
            logging.info(f"[MERTDataset] Loading cached waveforms from {cache_path}")
            self.waveforms_, self.wave_labels_ = torch.load(cache_path)
        else:
            logging.info(f"[MERTDataset] Precomputing waveforms for {len(self.index_)} tracks")
            self._precompute(cache_path, orig_sr)

    def _precompute(self, cache_path: Path, orig_sr: int):
        waveforms_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []

        for idx in tqdm(range(len(self.index_)), desc="Precomputing MERT waveforms"):
            try:
                audio_loader, label, _ = super().__getitem__(idx)
                audio = audio_loader("cpu").numpy()
                waveform = load_audio_mert(audio, orig_sr)
                waveforms_list.append(waveform)
                labels_list.append(label)
            except Exception as e:
                logging.warning(f"[MERTDataset] Skipping track {idx}: {e}")

        self.waveforms_ = waveforms_list
        self.wave_labels_ = torch.stack(labels_list)

        torch.save((self.waveforms_, self.wave_labels_), cache_path)
        logging.info(f"[MERTDataset] Saved waveform cache to {cache_path}")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return self.waveforms_[index], self.wave_labels_[index], index

    def __len__(self):
        return len(self.waveforms_)


def collate_mert(batch):
    """
    Pad waveforms to same length within batch.

    Returns:
        waveforms   : (B, T)        padded float32
        attention_mask: (B, T)      1 where real, 0 where padded
        labels      : (B,)          LongTensor
        ids         : list[int]
    """
    waveforms, labels, ids = zip(*batch)
    max_len = max(w.shape[0] for w in waveforms)

    padded = torch.zeros(len(waveforms), max_len)
    mask = torch.zeros(len(waveforms), max_len)

    for i, w in enumerate(waveforms):
        padded[i, :w.shape[0]] = w
        mask[i, :w.shape[0]] = 1.0

    return (
        padded,
        mask,
        torch.stack(labels).long(),
        list(ids),
    )