import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import numpy as np

from src.fma.fma_dataset import VariableFMADataset


class TrainingVisualizer:
	def __init__(self, dataset: VariableFMADataset, idstr: str, base_dir: Path = Path('analysis') / 'model' / 'validation'):
		self.dataset = dataset
		self.idstr = idstr
		self.dir = base_dir / idstr
		self.dir.mkdir(parents=True, exist_ok=True)
		self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'macro_f1': [], 'per_class_acc': []}
		self.class_names = [dataset.genre_encoder.inverse_transform([i])[0] for i in range(dataset.num_classes)]

	def update(self, epoch, train_loss, train_acc, val_loss, val_acc, macro_f1, per_class_acc, cm=None, snapshot=False):
		self.history['train_loss'].append(float(train_loss))
		self.history['val_loss'].append(float(val_loss))
		self.history['train_acc'].append(float(train_acc))
		self.history['val_acc'].append(float(val_acc))
		self.history['macro_f1'].append(float(macro_f1))
		self.history['per_class_acc'].append([float(a) for a in per_class_acc])
		ep_dir = self.dir
		if cm is not None:
			plt.figure(figsize=(10, 8))
			sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.class_names, yticklabels=self.class_names, cmap='Blues')
			plt.title(f'Epoch {epoch} Confusion Matrix (Latest)')
			plt.xlabel('Predicted')
			plt.ylabel('True')
			plt.tight_layout()
			plt.savefig(ep_dir / 'confusion_matrix_latest.png')
			plt.close()
		if snapshot and cm is not None:
			plt.figure(figsize=(10, 8))
			sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.class_names, yticklabels=self.class_names, cmap='Blues')
			plt.title(f'Epoch {epoch} Confusion Matrix Snapshot')
			plt.xlabel('Predicted')
			plt.ylabel('True')
			plt.tight_layout()
			plt.savefig(ep_dir / f'confusion_matrix_epoch_{epoch}.png')
			plt.close()
		plt.figure(figsize=(10, 6))
		for i, name in enumerate(self.class_names):
			acc_curve = [epoch_acc[i] for epoch_acc in self.history['per_class_acc']]
			plt.plot(range(len(acc_curve)), acc_curve, label=name)
		plt.title('Per-Class Accuracy Over Epochs')
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.legend()
		plt.tight_layout()
		plt.savefig(ep_dir / 'per_class_accuracy.png')
		plt.close()
		plt.figure(figsize=(10, 6))
		plt.plot(self.history['train_loss'], label='Train Loss')
		plt.plot(self.history['val_loss'], label='Val Loss')
		plt.title('Loss Over Epochs')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend()
		plt.tight_layout()
		plt.savefig(ep_dir / 'loss.png')
		plt.close()
		plt.figure(figsize=(10, 6))
		plt.plot(self.history['train_acc'], label='Train Accuracy')
		plt.plot(self.history['val_acc'], label='Val Accuracy')
		plt.plot(self.history['macro_f1'], label='Macro F1')
		plt.title('Accuracy and Macro F1 Over Epochs')
		plt.xlabel('Epoch')
		plt.ylabel('Score')
		plt.legend()
		plt.tight_layout()
		plt.savefig(ep_dir / 'accuracy_f1.png')
		plt.close()