"""
A very simple, useless example of a model using this framework
"""
import torch
import torch.nn as nn

from src.fma import *
from src.common import *
from src.constants import *


class ExampleFMAModel(AbstractFMAGenreModule):
	@classmethod
	def train_generic(cls, train_dataset, val_dataset):
		model = ExampleFMAModel()
		model.fma_train(train_dataset, val_dataset, batch_size=4, num_epochs=100)

	@classmethod
	def name(cls):
		return 'example'

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.fc_ = nn.Linear(64, NUM_CLASSES)

	def forward(self, batch_X: List[torch.FloatTensor]):
		features = torch.stack([x[:64] for x in batch_X], dim=0).to(batch_X[0].device)
		logits = self.fc_(features)
		return logits
