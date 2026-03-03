import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchvggish import vggish
from tqdm import tqdm
from functools import partial
from src.common import AbstractFMAGenreModule

# ----------------------------
# Collate function
# ----------------------------
def collate_fma(batch, device):
    """
    Collate function for FMA datasets. Handles SegmentLoader or raw tensors.
    """
    if isinstance(batch[0][0], torch.Tensor):
        audios = [t[0] for t in batch]
        labels = torch.tensor([t[1] for t in batch], dtype=torch.long)
    else:
        audios = [t[0](device) for t in batch]
        labels = torch.tensor([t[1] for t in batch], dtype=torch.long)
    return audios, labels

# ----------------------------
# VGGish model for FMA
# ----------------------------
class VGGishFMA(AbstractFMAGenreModule):
    MODEL_PATH = "vggish_fma_best.pt"

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # VGGish backbone
        self.model = vggish().to(self.device)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 16)
        ).to(self.device)

        # Audio transforms
        self.target_sr = 16000
        self.resampler = torchaudio.transforms.Resample(22050, self.target_sr)
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=400,
            hop_length=160,
            n_mels=64
        )

    @staticmethod
    def name():
        return "vggish"

    # ----------------------------
    # Convert waveform to VGGish examples
    # ----------------------------
    def waveform_to_examples(self, waveforms: list, srs: list = None):
        if srs is None:
            srs = [22050] * len(waveforms)

        batch_examples = []
        batch_sizes = []
        frame_size = 96  # VGGish expects 96-frame patches

        for waveform, sr in zip(waveforms, srs):
            # Resample if needed
            if sr != self.target_sr:
                waveform = self.resampler(waveform.unsqueeze(0)).squeeze(0)

            # Ensure waveform is long enough for MelSpectrogram
            min_len = self.mel_spec.n_fft + (frame_size - 1) * self.mel_spec.hop_length
            if waveform.numel() < min_len:
                pad_size = min_len - waveform.numel()
                waveform = torch.nn.functional.pad(waveform, (0, pad_size))

            # Compute log-Mel spectrogram
            mel = self.mel_spec(waveform.unsqueeze(0))
            log_mel = torch.log(mel + 1e-6).squeeze(0).transpose(0, 1)  # (time, n_mels)

            # Pad frames if too short
            if log_mel.size(0) < frame_size:
                pad_frames = frame_size - log_mel.size(0)
                pad = torch.zeros(pad_frames, log_mel.size(1), device=log_mel.device)
                log_mel = torch.cat([log_mel, pad], dim=0)

            # Split into 96-frame examples
            num_examples = log_mel.size(0) // frame_size
            examples = log_mel[:num_examples*frame_size].unfold(0, frame_size, frame_size)
            if examples.dim() == 2:
                examples = examples.unsqueeze(0)
            examples = examples.unsqueeze(1)  # (N, 1, 96, 64)

            batch_examples.append(examples)
            batch_sizes.append(examples.size(0))

        all_examples = torch.cat(batch_examples, dim=0).to(self.device)
        return all_examples, batch_sizes

    # ----------------------------
    # Forward pass
    # ----------------------------
    def forward(self, batch):
        all_examples, batch_sizes = self.waveform_to_examples(batch)
        embeddings = self.model(all_examples)

        all_embeddings = []
        start = 0
        for size in batch_sizes:
            emb = embeddings[start:start + size].mean(dim=0)
            all_embeddings.append(emb)
            start += size

        all_embeddings = torch.stack(all_embeddings)
        logits = self.classifier(all_embeddings)
        return logits

    # ----------------------------
    # Training tried to implement precomputing but the input would be wrong
    # ----------------------------
    def train_generic(self, train_dataset, val_dataset, batch_size=16, epochs=5):
        device = self.device

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=partial(collate_fma, device=device),
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=partial(collate_fma, device=device),
            num_workers=4,
            pin_memory=True
        )

        optimizer = torch.optim.Adam([
            {"params": self.model.parameters(), "lr": 1e-5},
            {"params": self.classifier.parameters(), "lr": 1e-3}
        ])
        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.0

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0

            for audios, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = self.forward(audios)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)

            # Validation
            self.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for audios, labels in val_loader:
                    labels = labels.to(device)
                    logits = self.forward(audios)
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += len(labels)

            val_acc = correct / total
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.state_dict(), self.MODEL_PATH)
                print("New best model saved.")

        print("Training complete.")

    # ----------------------------
    # Testing
    # ----------------------------
    def test_generic(self, test_dataset, batch_size=16):
        device = self.device
        self.load_state_dict(torch.load(self.MODEL_PATH, map_location=device))
        self.eval()

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=partial(collate_fma, device=device),
            num_workers=4,
            pin_memory=True
        )

        correct, total = 0, 0
        with torch.no_grad():
            for audios, labels in tqdm(test_loader, desc="Testing"):
                labels = labels.to(device)
                logits = self.forward(audios)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        print(f"Test accuracy: {correct / total:.4f}")