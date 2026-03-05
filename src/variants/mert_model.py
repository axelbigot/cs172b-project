"""
src/variants/mert_model.py

MERT-v1-95M backbone for FMA genre classification.
Optimized for A100 (40GB VRAM):
  - 30-second waveforms
  - batch_size=16
  - 6 unfrozen transformer layers with linear LR warmup
  - fp16 backbone
  - gradient accumulation x2 (effective batch_size=32)
  - early stopping patience=15
  - NaN detection
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
import logging

from src.common import AbstractFMAGenreModule
from src.model_analyzer import TrainingVisualizer
from src.constants import DATA_DIRECTORY
from src.fma.mert_dataset import MERTDataset, collate_mert

MERT_MODEL_NAME      = "m-a-p/MERT-v1-95M"
FREEZE_EPOCHS        = 5
UNFREEZE_LAYERS      = 6
BACKBONE_LR_SCALE    = 0.001   # very conservative — prevents NaN on unfreeze
BACKBONE_WARMUP      = 3       # epochs to linearly ramp backbone LR from 0 to target
HIDDEN_DIM           = 768
GRAD_ACCUM_STEPS     = 2
EARLY_STOP_PATIENCE  = 15


# ---------------------------------------------------------------------------
# Attention pooling
# ---------------------------------------------------------------------------
class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        scores = self.v(torch.tanh(self.W(x.float())))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        weights = torch.softmax(scores, dim=0)
        return (weights * x.float()).sum(dim=0)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def _add_gaussian_noise(waveforms: torch.Tensor, std: float = 0.005) -> torch.Tensor:
    return waveforms + torch.randn_like(waveforms) * std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MERTFMA(AbstractFMAGenreModule):
    def __init__(self, tag=""):
        super().__init__()
        self.tag = tag
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logging.info(f"[MERT] Loading {MERT_MODEL_NAME} in fp16")
        self.backbone = AutoModel.from_pretrained(
            MERT_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(self.device)

        self.backbone.gradient_checkpointing_enable()

        self.attn_pool = AttentionPool(HIDDEN_DIM).to(self.device)
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 512),
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
        return "mert"

    def forward(self, waveforms: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_values=waveforms.half(),
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden = outputs.last_hidden_state  # (B, T', 768)

        T_out = hidden.shape[1]
        attn_down = torch.nn.functional.interpolate(
            attention_mask.float().unsqueeze(1),
            size=T_out, mode='nearest'
        ).squeeze(1)

        track_embeddings = []
        for i in range(hidden.shape[0]):
            track_embeddings.append(self.attn_pool(hidden[i], attn_down[i]))
        track_embeddings = torch.stack(track_embeddings)

        return self.classifier(track_embeddings)

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

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _partial_unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        encoder_layers = self.backbone.encoder.layers
        for layer in encoder_layers[-UNFREEZE_LAYERS:]:
            for param in layer.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, 'layer_norm'):
            for param in self.backbone.layer_norm.parameters():
                param.requires_grad = True
        n_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        logging.info(f"[MERT] Partial unfreeze: last {UNFREEZE_LAYERS} layers, {n_trainable:,} backbone params trainable")

    def _make_optimizer(self, lr: float, backbone_frozen: bool):
        head_params = list(self.classifier.parameters()) + list(self.attn_pool.parameters())
        if backbone_frozen:
            return torch.optim.AdamW(head_params, lr=lr, weight_decay=1e-4)
        backbone_trainable = [p for p in self.backbone.parameters() if p.requires_grad]
        return torch.optim.AdamW([
            {'params': backbone_trainable, 'lr': 0.0},  # start at 0, warmed up manually
            {'params': head_params,        'lr': lr},
        ], weight_decay=1e-4)

    def fma_train(self, train_dataset, val_dataset, epochs=100, batch_size=16, lr=1e-4):
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_mert, num_workers=4, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_mert, num_workers=4, pin_memory=True,
        )

        criterion = nn.CrossEntropyLoss()
        idstr = self.get_idstr(train_dataset)
        visualizer = TrainingVisualizer(train_dataset, idstr=idstr)
        checkpoint_path = Path(DATA_DIRECTORY) / f"{idstr}-checkpoint.pt"
        best_checkpoint_path = Path(DATA_DIRECTORY) / f"{idstr}-best.pt"
        best_val_acc = 0.0
        epochs_no_improve = 0
        unfreeze_epoch = FREEZE_EPOCHS + 1  # epoch when backbone was unfrozen
        target_backbone_lr = lr * BACKBONE_LR_SCALE

        logging.info("[MERT] Freezing backbone for first %d epochs", FREEZE_EPOCHS)
        self._freeze_backbone()
        optimizer = self._make_optimizer(lr, backbone_frozen=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=FREEZE_EPOCHS, eta_min=lr * 0.1
        )

        for epoch in range(1, epochs + 1):

            # Partial unfreeze at epoch 6
            if epoch == unfreeze_epoch:
                logging.info("[MERT] Partially unfreezing last %d transformer layers", UNFREEZE_LAYERS)
                self._partial_unfreeze_backbone()
                optimizer = self._make_optimizer(lr, backbone_frozen=False)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs - FREEZE_EPOCHS, eta_min=lr * 0.01
                )

            # Linear warmup for backbone LR over BACKBONE_WARMUP epochs after unfreeze
            if epoch >= unfreeze_epoch:
                warmup_progress = min(1.0, (epoch - unfreeze_epoch) / BACKBONE_WARMUP)
                current_backbone_lr = target_backbone_lr * warmup_progress
                optimizer.param_groups[0]['lr'] = current_backbone_lr
                logging.info(f"[MERT] Backbone LR: {current_backbone_lr:.2e} (warmup {warmup_progress:.0%})")

            # -------- Train --------
            self.backbone.train()
            self.classifier.train()
            self.attn_pool.train()
            train_loss, train_correct = 0.0, 0
            nan_detected = False

            optimizer.zero_grad()
            for step, (waveforms, masks, labels, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training")):
                waveforms = waveforms.to(self.device)
                masks     = masks.to(self.device)
                labels    = labels.to(self.device)

                waveforms = _add_gaussian_noise(waveforms)

                logits = self.forward(waveforms, masks)
                loss = criterion(logits, labels) / GRAD_ACCUM_STEPS

                if torch.isnan(loss):
                    logging.error("[MERT] NaN loss at epoch %d step %d — stopping", epoch, step)
                    nan_detected = True
                    break

                loss.backward()

                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss    += loss.item() * GRAD_ACCUM_STEPS * labels.size(0)
                train_correct += (logits.argmax(1) == labels).sum().item()

            if nan_detected:
                print(f"NaN detected at epoch {epoch} — stopping. Lower BACKBONE_LR_SCALE.")
                break

            scheduler.step()
            train_loss /= len(train_dataset)
            train_acc   = train_correct / len(train_dataset)

            # -------- Validation --------
            self.backbone.eval()
            self.classifier.eval()
            self.attn_pool.eval()
            val_loss, val_correct = 0.0, 0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for waveforms, masks, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                    waveforms = waveforms.to(self.device)
                    masks     = masks.to(self.device)
                    labels    = labels.to(self.device)

                    logits = self.forward(waveforms, masks)
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

            current_lr = optimizer.param_groups[-1]['lr']
            print(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Macro F1: {macro_f1:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            state = {
                'backbone_state_dict':   self.backbone.state_dict(),
                'classifier_state_dict': self.classifier.state_dict(),
                'attn_pool_state_dict':  self.attn_pool.state_dict(),
                'optimizer_state_dict':  optimizer.state_dict(),
                'epoch':                 epoch,
            }
            torch.save(state, checkpoint_path)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(state, best_checkpoint_path)
                logging.info("[MERT] New best val acc: %.4f", best_val_acc)
            else:
                epochs_no_improve += 1
                logging.info("[MERT] No improvement for %d/%d epochs", epochs_no_improve, EARLY_STOP_PATIENCE)
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    logging.info("[MERT] Early stopping at epoch %d", epoch)
                    print(f"Early stopping at epoch {epoch} — no improvement for {EARLY_STOP_PATIENCE} epochs")
                    break

        logging.info("[MERT] Best val acc: %.4f", best_val_acc)
        return visualizer

    @torch.no_grad()
    def fma_test(self, test_dataset, batch_size=16):
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_mert,
        )
        self.backbone.eval()
        self.classifier.eval()
        self.attn_pool.eval()
        correct, total = 0, 0

        for waveforms, masks, labels, _ in tqdm(test_loader, desc="Testing"):
            waveforms = waveforms.to(self.device)
            masks     = masks.to(self.device)
            labels    = labels.to(self.device)
            logits = self.forward(waveforms, masks)
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)

        accuracy = correct / total
        logging.info(f"[MERT TEST] Accuracy: {accuracy:.4f}")
        return accuracy