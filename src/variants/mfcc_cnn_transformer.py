import torch
import torch.nn as nn
import random
import math
from typing import List

from src.constants import *
from src.common import *

def mfcc_collate(batch: List[Tuple[torch.Tensor, torch.LongTensor, int]]):
    mfccs = torch.stack([x for x, _, _ in batch], dim=0)
    labels = torch.stack([y for _, y, _ in batch], dim=0)
    ids = [i for _, _, i in batch]
    return mfccs, labels, ids

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MFCC_TransformerCNNModel(AbstractFMAGenreModule):
    @classmethod
    def collate_fn(cls):
        return mfcc_collate
    
    @classmethod
    def train_generic(cls, train_dataset, val_dataset, **kwargs):
        model = cls(train_dataset.num_classes, **kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        
        model.fma_train(train_dataset, val_dataset, batch_size=32, num_epochs=75, optimizer=optimizer, criterion=criterion)

    @classmethod
    def test_generic(cls, test_dataset: VariableFMADataset, **kwargs):
        model = cls(test_dataset.num_classes, **kwargs)
        acc = model.fma_test(test_dataset)
        print(f'Test accuracy: {acc*100:.2f}%')

    @classmethod
    def name(cls):
        return 'mfcc-transformer-cnn'

    def apply_spec_augment(self, x, max_time_mask=80, max_freq_mask=16):
        b, c, f, t = x.shape

        # Frequency Masking
        f_mask = random.randint(0, max_freq_mask)
        if f_mask > 0:
            f0 = random.randint(0, f - f_mask)
            x[:, :, f0:f0 + f_mask, :] = 0

        # Time Masking
        t_mask = random.randint(0, max_time_mask)
        t0 = random.randint(0, t - t_mask)
        x[:, :, :, t0:t0 + t_mask] = 0

        return x

    def __init__(self, num_classes: int, conv_channels: tuple[int, ...] = (32, 64, 128), n_heads: int = 4, transformer_layers: int = 1, dropout_p: float = 0.3, **kwargs):
        super().__init__(**kwargs)

        # CNN Feature Extractor
        layers = []
        input_ch = 1
        for ch in conv_channels:
            layers += [
                nn.Conv2d(input_ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 1)),
                nn.Dropout2d(p=0.25)
            ]
            input_ch = ch
        self.feature_extractor = nn.Sequential(*layers)

        # Positional Encoding & Transformer
        # d_model matches the number of channels from the last CNN layer
        self.d_model = conv_channels[-1]
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=512,
            dropout=dropout_p,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Final Classification Head
        self.classifier = nn.Linear(self.d_model, num_classes)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, batch_X: torch.Tensor, ids: List[int] = None) -> torch.Tensor:
        # Move to device and ensure 4D: [Batch, 1, Freq, Time]
        x = batch_X.to(next(self.parameters()).device).clone()
        if x.ndim == 3:
            x = x.unsqueeze(1)

        if self.training:
            x = self.apply_spec_augment(x)

        # CNN Pass -> Output: [Batch, 256, F', T']
        features = self.feature_extractor(x)

        # Reshape for Transformer
        # We pool over Frequency (dim 2) to get a clean sequence over Time (dim 3)
        features = torch.mean(features, dim=2)  # Shape: [Batch, 256, Time]
        features = features.permute(0, 2, 1)    # Shape: [Batch, Time, 256]

        # Transformer Pass
        features = self.pos_encoder(features)
        transformed = self.transformer(features)

        # Global Pooling + Classifier
        # Aggregate the sequence info (mean over time dimension)
        out = torch.mean(transformed, dim=1)
        out = self.dropout(out)
        return self.classifier(out)
    