"""
src/variants/vggish_model.py

Improvements over baseline:
  1. Attention pooling   – replaces mean-pool; learns which windows matter per track
  2. Differential LRs   – backbone gets 10x smaller LR than classifier head
  3. CosineAnnealingLR  – replaces aggressive StepLR
  4. Augmentation       – Gaussian noise + random frame dropout during training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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

FREEZE_EPOCHS = 5          # epochs to train classifier only before unfreezing backbone
BACKBONE_LR_SCALE = 0.1    # backbone LR = base_lr * this


# ---------------------------------------------------------------------------
# Attention pooling module
# ---------------------------------------------------------------------------
class AttentionPool(nn.Module):
    """
    Soft-attention over T window embeddings → single track embedding.
    score_t = v^T tanh(W h_t)   then   out = sum_t softmax(score_t) * h_t
    """
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, dim)
        scores = self.v(torch.tanh(self.W(x)))   # (T, 1)
        weights = torch.softmax(scores, dim=0)    # (T, 1)
        return (weights * x).sum(dim=0)           # (dim,)


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------
def _add_gaussian_noise(frames: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return frames + torch.randn_like(frames) * std


def _random_frame_dropout(frames: torch.Tensor, track_sizes: list, p: float = 0.1):
    """Randomly zero-out entire windows with probability p (training only)."""
    mask = (torch.rand(frames.size(0), device=frames.device) > p).float()
    return frames * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), track_sizes


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class VGGishFMA(AbstractFMAGenreModule):
    def __init__(self, tag=""):
        super().__init__()
        self.tag = tag
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # VGGish backbone
        self.model = vggish().to(self.device)

        # Move PCA buffers to device if they exist
        for attr in ('_pca_matrix', '_pca_means'):
            if hasattr(self.model, attr):
                setattr(self.model, attr, getattr(self.model, attr).to(self.device))

        # Attention pooling (replaces mean-pool)
        self.attn_pool = AttentionPool(128).to(self.device)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 16),
        ).to(self.device)

    @classmethod
    def name(cls):
        return "vggish"

    # ----------------------------
    # Forward
    # ----------------------------
    def forward(self, frames: torch.Tensor, track_sizes: list) -> torch.Tensor:
        embeddings = self.model(frames)           # (total_windows, 128)

        # Attention-pool window embeddings back to one per track
        track_embeddings = []
        idx = 0
        for n in track_sizes:
            chunk = embeddings[idx:idx + n]       # (n, 128)
            track_embeddings.append(self.attn_pool(chunk))
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

    def _make_optimizer(self, lr: float, backbone_frozen: bool):
        """
        Build optimizer with differential LRs.
        When backbone is frozen, only train classifier + attention pool.
        When unfrozen, backbone gets BACKBONE_LR_SCALE * lr.
        """
        head_params = list(self.classifier.parameters()) + list(self.attn_pool.parameters())
        if backbone_frozen:
            return torch.optim.Adam(head_params, lr=lr)
        return torch.optim.Adam([
            {'params': self.model.parameters(),  'lr': lr * BACKBONE_LR_SCALE},
            {'params': head_params,               'lr': lr},
        ])

    # ----------------------------
    # Training
    # ----------------------------
    def fma_train(self, train_dataset, val_dataset, epochs=100, batch_size=32, lr=3e-4):
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_vggish_frames,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_vggish_frames,
        )

        criterion = nn.CrossEntropyLoss()
        idstr = self.get_idstr(train_dataset)
        visualizer = TrainingVisualizer(train_dataset, idstr=idstr)
        checkpoint_path = Path(DATA_DIRECTORY) / f"{idstr}-checkpoint.pt"
        best_val_acc = 0.0
        best_checkpoint_path = Path(DATA_DIRECTORY) / f"{idstr}-best.pt"

        # ---- Phase 1: freeze backbone ----
        logging.info("[TRAIN] Freezing VGGish backbone for first %d epochs", FREEZE_EPOCHS)
        for param in self.model.parameters():
            param.requires_grad = False

        backbone_frozen = True
        optimizer = self._make_optimizer(lr, backbone_frozen=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=FREEZE_EPOCHS, eta_min=lr * 0.1
        )

        for epoch in range(1, epochs + 1):

            # ---- Phase 2: unfreeze backbone ----
            if epoch == FREEZE_EPOCHS + 1:
                logging.info("[TRAIN] Unfreezing VGGish backbone (backbone LR = %.2e)", lr * BACKBONE_LR_SCALE)
                for param in self.model.parameters():
                    param.requires_grad = True
                backbone_frozen = False
                optimizer = self._make_optimizer(lr, backbone_frozen=False)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs - FREEZE_EPOCHS, eta_min=lr * 0.01
                )

            # -------- Train --------
            self.model.train()
            self.classifier.train()
            self.attn_pool.train()
            train_loss, train_correct = 0.0, 0

            for frames, labels, track_sizes, _ in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
                frames = frames.to(self.device)
                labels = labels.to(self.device)

                # Augmentation (training only)
                frames = _add_gaussian_noise(frames)
                frames, track_sizes = _random_frame_dropout(frames, track_sizes)

                optimizer.zero_grad()
                logits = self.forward(frames, track_sizes)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss    += loss.item() * labels.size(0)
                train_correct += (logits.argmax(1) == labels).sum().item()

            scheduler.step()
            train_loss /= len(train_dataset)
            train_acc   = train_correct / len(train_dataset)

            # -------- Validation --------
            self.model.eval()
            self.classifier.eval()
            self.attn_pool.eval()
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

            # Get current LR for logging (head LR)
            current_lr = optimizer.param_groups[-1]['lr']
            print(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Macro F1: {macro_f1:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # Save latest checkpoint
            state = {
                'model_state_dict':      self.model.state_dict(),
                'classifier_state_dict': self.classifier.state_dict(),
                'attn_pool_state_dict':  self.attn_pool.state_dict(),
                'optimizer_state_dict':  optimizer.state_dict(),
                'epoch':                 epoch,
            }
            torch.save(state, checkpoint_path)

            # Save best checkpoint separately
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(state, best_checkpoint_path)
                logging.info("[TRAIN] New best val acc: %.4f — saved to %s", best_val_acc, best_checkpoint_path)

        logging.info("[TRAIN] Best val acc across all epochs: %.4f", best_val_acc)
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
        self.attn_pool.eval()
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