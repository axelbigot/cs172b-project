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
		model.fma_train(train_dataset, val_dataset, batch_size=4, num_epochs=5)

	@classmethod
	def name(cls):
		return 'example'

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.fc_ = nn.Linear(518, NUM_CLASSES)

	def forward(self, track):
		x = track.features.view(track.features.size(0), -1)
		logits = self.fc_(x)
		return logits
