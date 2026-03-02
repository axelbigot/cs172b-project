import torch
import torch.nn as nn
from typing import List

from src.constants import *
from src.common import *


def mfcc_collate(batch: List[Tuple[torch.Tensor, torch.LongTensor, int]]):
    mfccs = torch.stack([x for x, _, _ in batch], dim=0)
    labels = torch.stack([y for _, y, _ in batch], dim=0)
    ids = [i for _, _, i in batch]
    return mfccs, labels, ids

class MFCC_CNNFMAModel(AbstractFMAGenreModule):
    @classmethod
    def collate_fn(cls):
        return mfcc_collate
    
    @classmethod
    def train_generic(cls, train_dataset, val_dataset, tag):
        model = cls(train_dataset.num_classes, tag=tag).fma_train(train_dataset, val_dataset, batch_size=16, num_epochs=750)

    @classmethod
    def test_generic(cls, test_dataset: VariableFMADataset):
        model = cls(test_dataset.num_classes, tag=cls.name())
        acc = model.fma_test(test_dataset)
        print(f'Test accuracy: {acc*100:.2f}%')

    @classmethod
    def name(cls):
        return 'mfcc-cnn'

    def __init__(self, num_classes: int, conv_channels: tuple[int, ...] = (16, 32, 64, 128), dropout_p: float = 0.3, **kwargs):
        super().__init__(**kwargs)

        self.classifier = nn.Linear(conv_channels[-1], num_classes)
        self.dropout = nn.Dropout(dropout_p)

        layers = []
        input_ch = 1

        for ch in conv_channels:
            layers += [
                nn.Conv2d(input_ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ]
            input_ch = ch

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, batch_X: torch.Tensor, ids: List[int] = None) -> torch.Tensor:
        x = batch_X.to(next(self.parameters()).device)
        features = self.feature_extractor(x).flatten(1)
        features = self.dropout(features)

        return self.classifier(features)
    