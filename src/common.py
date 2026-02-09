import torch
import torch.nn as nn
import logging
import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.fma import VariableFMADataset, FMATrack


@dataclass
class FMATrackBatch:
	"""_summary_ Batched FMATracks.

	Note that the `audios` member is a list of variable-length tensors and may need some preprocessing
	for certain models (in `forward`).
	"""
	audios: list[torch.Tensor]
	lengths: torch.Tensor
	track_ids: torch.Tensor
	genres: torch.Tensor
	features: torch.Tensor
	echonests: torch.Tensor

def fma_track_collate(batch: list[FMATrack]) -> FMATrackBatch:
	"""_summary_ Collation function for batching FMATracks.

	Parameters
	----------
	batch : list[FMATrack]
			_description_ Raw FMATracks in the batch.

	Returns
	-------
	FMATrackBatch
			_description_ Batched FMATracks.
	"""
	return FMATrackBatch(
		audios=[torch.tensor(b.audio) for b in batch],
		lengths=torch.tensor([b.length for b in batch], dtype=torch.float32),
		track_ids=torch.tensor([b.track_id for b in batch], dtype=torch.int32),
		genres=torch.tensor([b.genre for b in batch], dtype=torch.long),
		features=torch.from_numpy(np.stack([b.features for b in batch], axis=0)).float(),
    echonests=torch.from_numpy(np.stack([b.echonest for b in batch], axis=0)).float()
	)

class AbstractFMAGenreModule(nn.Module, ABC):
	@staticmethod
	@abstractmethod
	def train_generic(cls):
		"""_summary_ Static method that each FMA model must implement which will be called by the main program.
		This function should delegate to `fma_train`.
		"""
		pass

	@staticmethod
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
	def forward(self, track: FMATrackBatch) -> torch.Tensor:
		"""_summary_ Forward method of the model.

		Parameters
		----------
		track : FMATrackBatch
				_description_ Batched FMA tracks.

		Returns
		-------
		torch.Tensor
				_description_ Logits tensor.
		"""
		pass

	def fma_train(
		self,
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
		
		self.to(device)
		dataset = VariableFMADataset()

		if optimizer is None:
			optimizer = torch.optim.Adam(self.parameters(), lr=lr)

		if criterion is None:
			criterion = torch.nn.CrossEntropyLoss()

		loader = DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=True,
			collate_fn=fma_track_collate
		)

		for epoch in range(num_epochs):
			self.train()
			running_loss = 0.0

			for batch in tqdm(loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
				batch = FMATrackBatch(
					audios=[a.to(device) for a in batch.audios],
					lengths=batch.lengths.to(device),
					track_ids=batch.track_ids.to(device),
					genres=batch.genres.to(device),
					features=batch.features.to(device),
					echonests=batch.echonests.to(device)
				)

				optimizer.zero_grad()
				outputs = self(batch)
				loss = criterion(outputs, batch.genres)
				loss.backward()
				optimizer.step()

				running_loss += loss.item() * len(batch.track_ids)

			epoch_loss = running_loss / len(dataset)
			logging.info(f'Epoch {epoch + 1} Loss: {epoch_loss}')
