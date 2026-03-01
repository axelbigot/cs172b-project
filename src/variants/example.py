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
	def train_generic(cls, train_dataset, val_dataset, tag):
		model = ExampleFMAModel(train_dataset.num_classes, tag=tag)
		model.fma_train(train_dataset, val_dataset, batch_size=4, num_epochs=100)

	@classmethod
	def test_generic(cls, test_dataset):
		model = cls(test_dataset.num_classes)
		test_accuracy = model.fma_test(test_dataset)

		logging.info(f'Test accuracy: {(test_accuracy * 100):6f}%')

	@classmethod
	def name(cls):
		return 'example'

	def __init__(self, num_classes, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.fc_ = nn.Linear(64, num_classes)

	def forward(self, batch_X: List[callable], ids: List[int]):
		device = next(self.parameters()).device

		audios = [load_audio(device) for load_audio in batch_X]
		features = torch.stack([x[:64] for x in audios], dim=0).to(device)

		logits = self.fc_(features)
		return logits
