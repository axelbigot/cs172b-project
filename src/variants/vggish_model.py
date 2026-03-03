"""
src/variants/vggish_model.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvggish import vggish
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
import logging

from src.common import AbstractFMAGenreModule
from src.model_analyzer import TrainingVisualizer
from src.constants import DATA_DIRECTORY
from src.fma.vgg_dataset import VGGishFrameDataset, collate_vggish_frames


class VGGishFMA(AbstractFMAGenreModule):
    def __init__(self, tag=""):
        super().__init__(tag)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # VGGish backbone + classifier head
        self.model = vggish().to(self.device)

        # Move PCA buffers to device if they exist
        if hasattr(self.model, '_pca_matrix'):
            self.model._pca_matrix = self.model._pca_matrix.to(self.device)
        if hasattr(self.model, '_pca_means'):
            self.model._pca_means = self.model._pca_means.to(self.device)

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 16),
        ).to(self.device)

    @classmethod
    def name(cls):
        return "vggish"

    def forward(self, frames: torch.Tensor, track_sizes: list) -> torch.Tensor:
        embeddings = self.model(frames)         # (total_windows, 128)

        # Mean-pool window embeddings back to one per track
        track_embeddings = []
        idx = 0
        for n in track_sizes:
            track_embeddings.append(embeddings[idx:idx + n].mean(dim=0))
            idx += n
        track_embeddings = torch.stack(track_embeddings)  # (B, 128)

        return self.classifier(track_embeddings)           # (B, 16)

    # ----------------------------
    # AbstractFMAGenreModule overrides
    # ----------------------------
    @classmethod
    def train_generic(cls, train_dataset, val_dataset, **kwargs):
        model = cls(tag=kwargs.get('tag', ''))
        return model.fma_train(train_dataset, val_dataset, **kwargs)

    @classmethod
    def test_generic(cls, test_dataset, **kwargs):
        model = cls(tag=kwargs.get('tag', ''))
        return model.fma_test(test_dataset, **kwargs)

    def get_idstr(self, dataset):
        base_dataset = getattr(dataset, 'dataset', dataset)
        frac = getattr(base_dataset, 'dowsample_frac', 'NA')
        return (
            f'model_trained_{self.name()}'
            f'{f"_tag-{self.tag}" if len(self.tag) else ""}'
            f'_{base_dataset.__class__.__name__}_frac-{frac}'
        )

    # ----------------------------
    # Training
    # ----------------------------
    def fma_train(self, train_dataset, val_dataset, epochs=20, batch_size=32, lr=1e-3):
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_vggish_frames,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_vggish_frames,
        )

        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.classifier.parameters()), lr=lr
        )
        criterion = nn.CrossEntropyLoss()

        idstr = self.get_idstr(train_dataset)
        visualizer = TrainingVisualizer(train_dataset, idstr=idstr)
        checkpoint_path = Path(DATA_DIRECTORY) / f"{idstr}-checkpoint.pt"

        for epoch in range(1, epochs + 1):
            # -------- Train --------
            self.model.train()
            self.classifier.train()
            train_loss, train_correct = 0.0, 0

            for frames, labels, track_sizes, _ in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
                frames = frames.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.forward(frames, track_sizes)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss    += loss.item() * labels.size(0)
                train_correct += (logits.argmax(1) == labels).sum().item()

            train_loss /= len(train_dataset)
            train_acc   = train_correct / len(train_dataset)

            # -------- Validation --------
            self.model.eval()
            self.classifier.eval()
            val_loss, val_correct = 0.0, 0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for frames, labels, track_sizes, _ in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                    frames = frames.to(self.device)
                    labels = labels.to(self.device)

                    logits = self.forward(frames, track_sizes)
                    loss   = criterion(logits, labels)

                    val_loss    += loss.item() * labels.size(0)
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    all_preds.append(logits.argmax(1).cpu())
                    all_labels.append(labels.cpu())

            val_loss /= len(val_dataset)
            val_acc   = val_correct / len(val_dataset)

            all_preds  = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            macro_f1   = f1_score(all_labels, all_preds, average='macro')
            cm         = confusion_matrix(all_labels, all_preds)
            per_class_acc = cm.diagonal() / cm.sum(axis=1)

            visualizer.update(
                epoch, train_loss, train_acc, val_loss, val_acc,
                macro_f1, per_class_acc, cm=cm, snapshot=True,
            )

            print(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Macro F1: {macro_f1:.4f}"
            )

            torch.save({
                'model_state_dict':      self.model.state_dict(),
                'classifier_state_dict': self.classifier.state_dict(),
                'optimizer_state_dict':  optimizer.state_dict(),
                'epoch':                 epoch,
            }, checkpoint_path)

        return visualizer

    # ----------------------------
    # Testing
    # ----------------------------
    @torch.no_grad()
    def fma_test(self, test_dataset, batch_size=32):
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_vggish_frames,
        )
        self.model.eval()
        self.classifier.eval()
        correct, total = 0, 0

        for frames, labels, track_sizes, _ in tqdm(test_loader, desc="Testing"):
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            logits = self.forward(frames, track_sizes)
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)

        accuracy = correct / total
        logging.info(f"[TEST] Accuracy: {accuracy:.4f}")
        return accuracy