import torch
import torch.nn as nn
import numpy as np
import librosa
from typing import List

from src.constants import *
from src.common import *

class MFCC_CNNFMAModel(AbstractFMAGenreModule):
    @classmethod
    def train_generic(cls, train_dataset, val_dataset):
        model = cls().fma_train(train_dataset, val_dataset, batch_size = 16, num_epochs = 1500)
    
    @classmethod
    def name(cls):
        return 'mfcc-cnn'

    def __init__(self, sample_rate: int = 22050, mfcc_coeffs: int = 40, fft_window: int = 2048, hop_length: int = 512, conv_channels: tuple[int, ...] = (16, 32, 64, 128), **kwargs):
        super().__init__(**kwargs)

        self.sample_rate = sample_rate
        self.mfcc_coeffs = mfcc_coeffs
        self.fft_window = fft_window
        self.hop_length = hop_length

        # build CNN layers
        self.feature_extractor = self._build_cnn(conv_channels)
        self.classifier = nn.Linear(conv_channels[-1], NUM_CLASSES)

        self.mfcc_cache = {}
    
    def _build_cnn(self, channels: tuple[int, ...]) -> nn.Sequential:
        layers = []
        input_ch = 1

        for ch in channels:
            layers += [
                nn.Conv2d(input_ch, ch, 3, padding = 1, bias = False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(2)
            ]
            input_ch = ch

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        return nn.Sequential(*layers)
    
    def _mfcc(self, audio: np.ndarray, aid: int) -> np.ndarray:
        if aid in self.mfcc_cache:
            return self.mfcc_cache[aid]

        audio = audio.astype(np.float32) if audio.dtype != np.float32 else audio
        mfcc_feat = librosa.feature.mfcc(y = audio, sr = self.sample_rate, n_mfcc = self.mfcc_coeffs, n_fft = self.fft_window, hop_length = self.hop_length)

        # normalize
        mean, std = np.mean(mfcc_feat), np.std(mfcc_feat)
        std = std if std > 1e-6 else 1.0
        mfcc_norm = (mfcc_feat - mean) / std

        self.mfcc_cache[aid] = mfcc_norm

        return mfcc_norm
    
    def forward(self, batch_X: List[torch.Tensor], ids: List[int]) -> torch.Tensor:
        # convert batch to MFCCs and pad
        mfcc_tensors = []
        for audio, aid in zip(batch_X, ids):
            arr = audio.detach().cpu().numpy()
            if arr.ndim > 1:
                arr = arr.mean(axis = 0)
            mfcc_arr = self._mfcc(arr, aid)
            mfcc_tensors.append(torch.from_numpy(mfcc_arr)[None, :]) 

        # pad along time dimension
        max_len = max(t.shape[2] for t in mfcc_tensors)
        padded = [
            torch.cat([t, t.new_zeros((1, self.mfcc_coeffs, max_len - t.shape[2]))], dim = 2)
            if t.shape[2] < max_len else t
            for t in mfcc_tensors
        ]

        x = torch.stack(padded).to(next(self.parameters()).device)
        features = self.feature_extractor(x).flatten(1)

        return self.classifier(features)
