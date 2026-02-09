"""
A very simple, useless example of a model using this framework
"""
import torch
import torch.nn as nn

from src.fma import *
from src.common import *
from src.constants import *


class ExampleFMAModel(AbstractFMAGenreModule):
	@staticmethod
	def train_generic():
		model = ExampleFMAModel()
		model.fma_train(batch_size=4, num_epochs=1)

	@staticmethod
	def name():
		return 'example'

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.fc_ = nn.Linear(518, NUM_CLASSES)

	def forward(self, track):
		x = track.features.view(track.features.size(0), -1)
		logits = self.fc_(x)
		return logits
