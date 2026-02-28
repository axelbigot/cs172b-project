import torch
import torch.nn as nn
import logging
import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from os import PathLike
from typing import List, Tuple

from src.constants import *
from src.fma import VariableFMADataset


def audio_genre_collate(batch: List[Tuple[torch.FloatTensor, torch.LongTensor, int]]) -> Tuple[List[torch.FloatTensor], torch.LongTensor, List[int]]:
	audios = [audio for audio, _, _ in batch]
	labels = torch.stack([genre for _, genre, _ in batch])
	ids = [i for _, _, i in batch]

	return audios, labels, ids

class AbstractFMAGenreModule(nn.Module, ABC):
	@classmethod
	@abstractmethod
	def train_generic(cls, train_dataset: Subset, val_dataset: Subset):
		"""_summary_ Static method that each FMA model must implement which will be called by the main program.
		This function should delegate to `fma_train`.

		Parameters
		----------
		train_dataset : Subset
				_description_ Training dataset (train split)
		val_dataset : Subset
				_description_ Validation dataset (validation split)
		"""
		pass

	@classmethod
	def test_generic(cls: type['AbstractFMAGenreModule'], test_dataset: Subset):
		"""_summary_ Static method that delegates to the unified testing loop.

		Parameters
		----------
		test_dataset : Subset
				_description_ Test dataset (test split of data).
		"""
		model = cls()
		test_accuracy = model.fma_test(test_dataset)

		logging.info(f'Test accuracy: {(test_accuracy * 100):6f}%')

	@classmethod
	@abstractmethod
	def name(cls) -> str:
		"""_summary_ Static method that each FMA model must implement which returns the display name of the
		model. This will be used my the main program as the CLI arg to run this specific model.

		Returns
		-------
		str
				_description_ Display name (ideally short). This value must be unique compared to other models.
		"""
		pass
	
	@abstractmethod
	def forward(self, x: List[torch.Tensor], ids: List[int]) -> torch.Tensor:
		"""_summary_ Forward method of the model.

		Parameters
		----------
		x : List[torch.Tensor]
				_description_ Batched audio byte tensors (variable length).

		Returns
		-------
		torch.Tensor
				_description_ Logits tensor.
		"""
		pass

	def fma_train(
		self,
		train_dataset: Subset,
		val_dataset: Subset,
		batch_size: int=16,
		optimizer: torch.optim.Optimizer | None = None,
		criterion: torch.nn.Module | None = None,
		lr: float=1e-3,
		num_epochs: int=10,
		device: str='cuda' if torch.cuda.is_available() else 'cpu'
	):
		"""_summary_ Parameterized unified training loop for all models. This function should be invoked by
		the model's `train_generic`.

		Parameters
		----------
		train_dataset : Subset
				_description_, Training dataset (train split)
		val_dataset : Subset
				_description_, Validation dataset (validation split)
		batch_size : int, optional
				_description_, by default 16
		optimizer : torch.optim.Optimizer | None, optional
				_description_, by default None
		criterion : torch.nn.Module | None, optional
				_description_, by default None
		lr : float, optional
				_description_, by default 1e-3
		num_epochs : int, optional
				_description_, by default 10
		device : str, optional
				_description_, by default 'cuda'iftorch.cuda.is_available()else'cpu'
		"""
		logging.info(f'[TRAINING] Using device "{device}"')

		if optimizer is None:
			optimizer = torch.optim.Adam(self.parameters(), lr=lr)

		if criterion is None:
			criterion = torch.nn.CrossEntropyLoss()

		epoch = 0

		path = DATA_DIRECTORY / f'model_trained_{self.name()}'
		log_dir = DATA_DIRECTORY / f'runs/{self.name()}'

		log_dir.mkdir(parents=True, exist_ok=True)

		writer = SummaryWriter(log_dir=str(log_dir))

		if path.exists():
			cp = torch.load(path, map_location=device)

			self.load_state_dict(cp['model_state_dict'])
			optimizer.load_state_dict(cp['optimizer_state_dict'])

			for state in optimizer.state.values():
				for k, v in state.items():
					if isinstance(v, torch.Tensor):
						state[k] = v.to(device)

			epoch = cp['epoch']

		criterion.to(device)
		self.to(device)
		self.eval()

		train_loader = DataLoader(
			train_dataset,
			batch_size=batch_size,
			shuffle=True,
			collate_fn=audio_genre_collate
		)

		val_loader = DataLoader(
			val_dataset,
			batch_size=batch_size,
			shuffle=False,
			collate_fn=audio_genre_collate
		)

		for ep in tqdm(range(epoch, num_epochs), desc='Total Epochs'):
			epoch = ep

			self.train()
			train_loss = 0.0
			n_correct = 0
			total = 0

			for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
				batch_X, batch_y, ids = self.batch_to_device_(batch, device)

				optimizer.zero_grad()
				outputs = self(batch_X, ids)
				loss = criterion(outputs, batch_y)
				loss.backward()
				optimizer.step()

				train_loss += loss.item() * len(batch_y)
				preds = outputs.argmax(1)
				n_correct += (preds == batch_y).sum().item()
				total += len(batch_y)

			train_accuracy = n_correct / total
			train_loss /= total

			val_loss, val_accuracy = self.evaluate(val_loader, device, criterion)

			logging.info(
				f'Epoch {epoch+1}\n'
				f'\tTrain loss: {train_loss:6f}, accuracy: {(train_accuracy * 100):6f}%\n'
				f'\tValidation loss: {val_loss:6f}, accuracy: {(val_accuracy * 100):6f}%'
			)

			writer.add_scalar('Loss/Train', train_loss, ep)
			writer.add_scalar('Accuracy/Train', train_accuracy, ep)
			writer.add_scalar('Loss/Validation', val_loss, ep)
			writer.add_scalar('Accuracy/Validation', val_accuracy, ep)

			torch.save({
				'model_state_dict': self.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'epoch': epoch+1
			}, path)

	def fma_test(
		self, 
		test_dataset: Subset, 
		batch_size=16,
		device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
	) -> float:
		path = DATA_DIRECTORY / f'model_trained_{self.name()}'
		if path.exists():
			cp = torch.load(path, map_location=device)
			self.load_state_dict(cp['model_state_dict'])

		test_loader = DataLoader(
			test_dataset,
			shuffle=False,
			batch_size=batch_size,
			collate_fn=audio_genre_collate
		)
		
		self.to(device)
		self.eval()

		correct = 0
		total = 0

		for batch in tqdm(test_loader, desc=f'Testing in progress'):
			batch_X, batch_y, ids = self.batch_to_device_(batch, device)

			logits = self(batch_X, ids)
			preds = logits.argmax(dim=1)

			correct += (preds == batch_y).sum().item()
			total += batch_y.size(0)

		return correct / total

	@torch.no_grad()
	def evaluate(
		self, 
		loader: DataLoader, 
		device: str, 
		criterion: nn.Module
	) -> tuple[float, float]:
		self.eval()

		total_loss = 0.0
		n_correct = 0
		total = 0

		for batch in loader:
			batch_X, batch_y, ids = self.batch_to_device_(batch, device)
			outputs = self(batch_X, ids)

			loss = criterion(outputs, batch_y)
			total_loss += loss.item() * len(batch_y)

			preds = outputs.argmax(dim=1)
			n_correct += (preds == batch_y).sum().item()
			total += len(batch_y)

		accuracy = n_correct / total
		avg_loss = total_loss / total

		return avg_loss, accuracy

	def batch_to_device_(self, batch: Tuple[List[torch.FloatTensor], torch.LongTensor, List[int]], device: str) -> Tuple[List[torch.FloatTensor], torch.LongTensor, List[int]]:
		audios = [x.to(device) for x in batch[0]]
		return (audios, batch[1].to(device), batch[2])
