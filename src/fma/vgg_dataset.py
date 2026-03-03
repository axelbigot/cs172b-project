"""
src/variants/vggish_dataset.py

Precomputes VGGish-ready frames (N, 1, 64, 96) from FMA audio and caches to disk.
First run is slow; all subsequent runs load instantly from cache.
"""
from src.fma.fma_dataset import VariableFMADataset
from pathlib import Path
import torch
import numpy as np
import librosa
import logging
from typing import Tuple, List
from tqdm import tqdm
from src.constants import DATA_DIRECTORY, RANDOM_SEED


# ----------------------------
# VGGish constants
# ----------------------------
VGGISH_SAMPLE_RATE  = 16000
VGGISH_N_MELS       = 64
VGGISH_N_FFT        = 400
VGGISH_HOP_LENGTH   = 160   # 10ms at 16kHz
VGGISH_FRAME_LENGTH = 96    # frames per window (0.96s)


def audio_to_vggish_frames(audio: np.ndarray, orig_sr: int) -> torch.Tensor:
    """
    Convert a raw audio array to VGGish-compatible frames.

    Steps:
      1. Resample to 16kHz
      2. Compute log-mel spectrogram  -> (64, T)
      3. Slice into non-overlapping 96-frame windows
      4. Return tensor of shape (N, 1, 64, 96)
    """
    # 1. Resample
    if orig_sr != VGGISH_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=VGGISH_SAMPLE_RATE)

    # 2. Log-mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=VGGISH_SAMPLE_RATE,
        n_fft=VGGISH_N_FFT,
        hop_length=VGGISH_HOP_LENGTH,
        n_mels=VGGISH_N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)  # (64, T)

    # 3. Slice into 96-frame windows
    T = log_mel.shape[1]
    n_windows = max(1, T // VGGISH_FRAME_LENGTH)
    log_mel = log_mel[:, :n_windows * VGGISH_FRAME_LENGTH]       # (64, n_windows*96)
    frames = log_mel.reshape(VGGISH_N_MELS, n_windows, VGGISH_FRAME_LENGTH)  # (64, N, 96)
    frames = frames.transpose(1, 0, 2)                            # (N, 64, 96)

    # 4. Add channel dim -> (N, 1, 64, 96)
    return torch.from_numpy(frames.astype(np.float32)).unsqueeze(1)


class VGGishFrameDataset(VariableFMADataset):
    """
    Extends VariableFMADataset by precomputing VGGish frames for every track
    and caching them to disk.

    __getitem__ returns:
        frames : (N, 1, 64, 96)  — all VGGish windows for the track
        label  : ()  LongTensor
        index  : int
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        orig_sr = self.sampling_rate_
        cache_path = (
            Path(DATA_DIRECTORY)
            / f"vggish_frames_cache_{self.idstr_}.pt"
        )

        if cache_path.exists():
            logging.info(f"[VGGishFrameDataset] Loading cached frames from {cache_path}")
            self.frames_: List[torch.Tensor]
            self.frame_labels_: torch.Tensor
            self.frames_, self.frame_labels_ = torch.load(cache_path)
        else:
            logging.info(f"[VGGishFrameDataset] Precomputing VGGish frames for {len(self.index_)} tracks")
            self._precompute(cache_path, orig_sr)

    def _precompute(self, cache_path: Path, orig_sr: int):
        frames_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []

        for idx in tqdm(range(len(self.index_)), desc="Precomputing VGGish frames"):
            audio_loader, label, _ = super().__getitem__(idx)
            audio = audio_loader("cpu").numpy()

            frames = audio_to_vggish_frames(audio, orig_sr)  # (N, 1, 64, 96)
            frames_list.append(frames)
            labels_list.append(label)

        self.frames_ = frames_list
        self.frame_labels_ = torch.stack(labels_list)  # (total_tracks,)

        torch.save((self.frames_, self.frame_labels_), cache_path)
        logging.info(f"[VGGishFrameDataset] Saved frame cache to {cache_path}")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if not hasattr(self, 'frames_'):
            raise RuntimeError("frames_ not initialized — precompute likely failed")
        return self.frames_[index], self.frame_labels_[index], index

    def __len__(self):
        return len(self.frames_)


def collate_vggish_frames(batch):
    """
    Collate a batch of (frames, label, idx) tuples.

    Returns:
        all_frames  : (total_windows, 1, 64, 96)
        labels      : (B,)
        track_sizes : list[int] — number of windows per track
        ids         : list[int]
    """
    all_frames  = []
    track_sizes = []
    labels      = []
    ids         = []

    for frames, label, idx in batch:
        all_frames.append(frames)           # (N, 1, 64, 96)
        track_sizes.append(frames.shape[0])
        labels.append(label)
        ids.append(idx)

    return (
        torch.cat(all_frames, dim=0),   # (sum_N, 1, 64, 96)
        torch.stack(labels).long(),     # (B,)
        track_sizes,
        ids,
    )