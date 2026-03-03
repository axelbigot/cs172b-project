import torch
import torch.nn as nn
import random
from typing import List

from src.constants import *
from src.common import *

def mfcc_collate(batch: List[Tuple[torch.Tensor, torch.LongTensor, int]]):
    mfccs = torch.stack([x for x, _, _ in batch], dim=0)
    labels = torch.stack([y for _, y, _ in batch], dim=0)
    ids = [i for _, _, i in batch]
    return mfccs, labels, ids

class MFCC_CNNFMAModelV3(AbstractFMAGenreModule):
    @classmethod
    def collate_fn(cls):
        return mfcc_collate
    
    @classmethod
    def train_generic(cls, train_dataset, val_dataset, **kwargs):
        model = cls(train_dataset.num_classes, **kwargs)

        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr = 5e-4, 
            weight_decay = 1e-3  
        )
        
        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing = 0.1 
        )

        model.fma_train(train_dataset, val_dataset, batch_size = 32, num_epochs = 75, optimizer = optimizer, criterion = criterion)

    @classmethod
    def test_generic(cls, test_dataset: VariableFMADataset, **kwargs):
        model = cls(test_dataset.num_classes, **kwargs)
        acc = model.fma_test(test_dataset)
        print(f'Test accuracy: {acc*100:.2f}%')

    @classmethod
    def name(cls):
        return 'mfcc-cnn-v3'

    def apply_spec_augment(self, x, max_time_mask=50, max_freq_mask=10):
        b, c, f, t = x.shape

        # Frequency Masking
        f_mask = random.randint(0, max_freq_mask)
        f0 = random.randint(0, f - f_mask)
        x[:, :, f0:f0 + f_mask, :] = 0
        
        # Time Masking
        t_mask = random.randint(0, max_time_mask)
        t0 = random.randint(0, t - t_mask)
        x[:, :, :, t0:t0 + t_mask] = 0
        
        return x

    def __init__(self, num_classes: int, conv_channels: tuple[int, ...] = (32, 64, 128, 256), dropout_p: float = 0.4, **kwargs):
        super().__init__(**kwargs)

        layers = []
        input_ch = 1

        for ch in conv_channels:
            layers += [
                nn.Conv2d(input_ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(p=0.2)
            ]
            input_ch = ch

        self.feature_extractor = nn.Sequential(*layers)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.classifier = nn.Linear(conv_channels[-1] * 2, num_classes)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, batch_X: torch.Tensor, ids: List[int] = None) -> torch.Tensor:
        x = batch_X.to(next(self.parameters()).device).clone()

        if x.ndim == 3:
            x = x.unsqueeze(1)

        if self.training:
            x = self.apply_spec_augment(x)

        features = self.feature_extractor(x)

        avg_f = self.avg_pool(features).flatten(1)
        max_f = self.max_pool(features).flatten(1)
        combined = torch.cat([avg_f, max_f], dim=1)

        combined = self.dropout(combined)
        return self.classifier(combined)
    