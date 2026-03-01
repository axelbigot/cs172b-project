import torch
import torch.nn as nn
import logging
import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from os import PathLike
from typing import List, Tuple
from sklearn.metrics import confusion_matrix, f1_score

from src.constants import *
from src.fma import VariableFMADataset


def audio_genre_collate(batch: List[Tuple[callable, torch.LongTensor, int]]) -> Tuple[List[callable], torch.LongTensor, List[int]]:
	load_audios = [load_audio for load_audio, _, _ in batch]
	labels = torch.stack([genre for _, genre, _ in batch])
	ids = [i for _, _, i in batch]

	return load_audios, labels, ids

class AbstractFMAGenreModule(nn.Module, ABC):
	@classmethod
	@abstractmethod
	def train_generic(cls, train_dataset: VariableFMADataset, val_dataset: VariableFMADataset, tag: str):
		"""_summary_ Static method that each FMA model must implement which will be called by the main program.
		This function should delegate to `fma_train`.

		Parameters
		----------
		train_dataset : VariableFMADataset
				_description_ Training dataset (train split)
		val_dataset : VariableFMADataset
				_description_ Validation dataset (validation split)
		"""
		pass

	@classmethod
	@abstractmethod
	def test_generic(cls: type['AbstractFMAGenreModule'], test_dataset: VariableFMADataset):
		"""_summary_ Static method that delegates to the unified testing loop.

		Parameters
		----------
		test_dataset : VariableFMADataset
				_description_ Test dataset (test split of data).
		"""

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

	@classmethod
	def collate_fn(cls):
		return audio_genre_collate
	
	def __init__(self, tag: str):
		self.tag = tag
	
	@abstractmethod
	def forward(self, x: List[callable], ids: List[int]) -> torch.Tensor:
		"""_summary_ Forward method of the model.

		Parameters
		----------
		x : List[callable]
				_description_ returns batched audio byte tensors (variable length).

		Returns
		-------
		torch.Tensor
				_description_ Logits tensor.
		"""
		pass

	def transform_batch(self, dataset, x, ids):
		return x

	def fma_train(
		self,
		train_dataset: VariableFMADataset,
		val_dataset: VariableFMADataset,
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
		train_dataset : VariableFMADataset
				_description_, Training dataset (train split)
		val_dataset : VariableFMADataset
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

		path = DATA_DIRECTORY / f'model_trained_{self.name()}{f"_tag-{self.tag}" if len(self.tag) else ""}_{train_dataset.__class__.__name__}_frac-{train_dataset.dowsample_frac}'
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
			collate_fn=self.collate_fn(),
		)

		val_loader = DataLoader(
			val_dataset,
			batch_size=batch_size,
			shuffle=False,
			collate_fn=self.collate_fn(),
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
				outputs = self(self.transform_batch(train_dataset, batch_X, ids), ids)
				loss = criterion(outputs, batch_y)
				loss.backward()
				optimizer.step()

				train_loss += loss.item() * len(batch_y)
				preds = outputs.argmax(1)
				n_correct += (preds == batch_y).sum().item()
				total += len(batch_y)

			train_accuracy = n_correct / total
			train_loss /= total

			val_loss, val_accuracy, val_preds, val_labels = self.evaluate(
				val_dataset,
				val_loader,
				device,
				criterion,
				return_preds=True
			)

			macro_f1 = f1_score(val_labels.numpy(), val_preds.numpy(), average='macro')

			logging.info(
				f'Epoch {epoch+1}\n'
				f'\tTrain loss: {train_loss:6f}, accuracy: {(train_accuracy * 100):6f}%\n'
				f'\tValidation loss: {val_loss:6f}, accuracy: {(val_accuracy * 100):6f}%\n'
				f"Validation Macro F1: {macro_f1:.4f}"
			)

			cm = confusion_matrix(val_labels.numpy(), val_preds.numpy())
			print(f'\n{cm}')

			per_class_acc = cm.diagonal() / cm.sum(axis=1)
			for i, acc in enumerate(per_class_acc):
					class_name = val_dataset.genre_encoder.inverse_transform([i])[0]
					logging.info(f"Class {class_name}: {acc*100:.2f}%")
					writer.add_scalar(f'PerClassAcc/{class_name}', acc, ep)

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
		test_dataset: VariableFMADataset, 
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
			collate_fn=self.collate_fn()
		)
		
		self.to(device)
		self.eval()

		correct = 0
		total = 0

		for batch in tqdm(test_loader, desc=f'Testing in progress'):
			batch_X, batch_y, ids = self.batch_to_device_(batch, device)

			logits = self(self.transform_batch(test_dataset, batch_X, ids), ids)
			preds = logits.argmax(dim=1)

			correct += (preds == batch_y).sum().item()
			total += batch_y.size(0)

		return correct / total

	@torch.no_grad()
	def evaluate(
		self, 
		dataset: VariableFMADataset,
		loader: DataLoader, 
		device: str, 
		criterion: nn.Module,
		return_preds: bool = False
	):
		self.eval()

		total_loss = 0.0
		n_correct = 0
		total = 0

		all_preds = []
		all_labels = []

		for batch in loader:
			batch_X, batch_y, ids = self.batch_to_device_(batch, device)
			outputs = self(self.transform_batch(dataset, batch_X, ids), ids)

			loss = criterion(outputs, batch_y)
			total_loss += loss.item() * len(batch_y)

			preds = outputs.argmax(dim=1)

			n_correct += (preds == batch_y).sum().item()
			total += len(batch_y)

			if return_preds:
				all_preds.append(preds.cpu())
				all_labels.append(batch_y.cpu())

		accuracy = n_correct / total
		avg_loss = total_loss / total

		if return_preds:
			return avg_loss, accuracy, torch.cat(all_preds), torch.cat(all_labels)

		return avg_loss, accuracy

	def batch_to_device_(self, batch: Tuple[List[torch.FloatTensor], torch.LongTensor, List[int]], device: str) -> Tuple[List[callable], torch.LongTensor, List[int]]:
		return (batch[0], batch[1].to(device), batch[2])
