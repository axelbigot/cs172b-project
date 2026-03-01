# src/variants/vggish_model.py
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from torchvggish import vggish
from tqdm import tqdm
from src.common import AbstractFMAGenreModule


# ----------------------------
# way to process the dataloader
# ----------------------------
def collate_fma(batch):
    """
    Returns:
        audios: list of 1D torch tensors (variable lengths)
        labels: torch tensor of genre labels
    """
    audios = [torch.tensor(t.audio, dtype=torch.float32) for t in batch]
    labels = torch.tensor([t.genre for t in batch], dtype=torch.long)
    return audios, labels


# ----------------------------
# Preprocessing for the Tensors
# ----------------------------
def waveform_to_examples_tensor(waveform: torch.Tensor, sr: int = 22050):
    """
    Converts 1D waveform tensor to VGGish examples tensor.
    Returns: [num_examples, 1, 96, 64]
    """
    device = waveform.device
    target_sr = 16000

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr).to(device)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=400,
        hop_length=160,
        n_mels=64
    ).to(device)

    mel = mel_spec(waveform.unsqueeze(0))
    log_mel = torch.log(mel + 1e-6).transpose(1, 2)

    frame_size = 96
    examples = []

    for start in range(0, log_mel.size(1) - frame_size + 1, frame_size):
        examples.append(log_mel[:, start:start + frame_size, :])

    if not examples:
        pad = torch.zeros(1, frame_size, 64, device=device)
        examples = [pad]

    examples_tensor = torch.cat(examples, dim=0)
    examples_tensor = examples_tensor.unsqueeze(1)
    return examples_tensor


# ----------------------------
# VGG model for FMA
# ----------------------------
class VGGishFMA(AbstractFMAGenreModule):

    MODEL_PATH = "vggish_fma_best.pt"

    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = vggish()
        self.model.eval()  # frozen feature extractor
        self.model.to(self.device)

        self.classifier = nn.Linear(128, 16)
        self.classifier.to(self.device)

    @staticmethod
    def name():
        return "vggish"

    def forward(self, batch):

        device = self.device
        all_embeddings = []
        batch_sizes = []
        examples_list = []

        for audio in batch:
            audio = audio.float().to(device)
            examples = waveform_to_examples_tensor(audio, sr=22050).to(device)
            examples_list.append(examples)
            batch_sizes.append(examples.size(0))

        all_examples = torch.cat(examples_list, dim=0)

        with torch.no_grad():
            embeddings = self.model(all_examples)

        start = 0
        for size in batch_sizes:
            emb = embeddings[start:start + size].mean(dim=0)
            all_embeddings.append(emb)
            start += size

        all_embeddings = torch.stack(all_embeddings)
        logits = self.classifier(all_embeddings)

        return logits

    # ----------------------------
    # Training Loop
    # ----------------------------
    @staticmethod
    def train_generic(train_dataset, val_dataset, batch_size=2, epochs=25):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = VGGishFMA().to(device)

        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fma
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_fma
        )

        best_val_acc = 0.0

        for epoch in tqdm(range(epochs), desc="Training Progress"):

            model.classifier.train()
            running_loss = 0.0

            train_bar = tqdm(train_loader,
                             desc=f"Epoch [{epoch+1}/{epochs}]",
                             leave=False)

            for audios, labels in train_bar:

                labels = labels.to(device)
                optimizer.zero_grad()

                logits = model.forward(audios)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                train_bar.set_postfix(loss=loss.item())

            avg_loss = running_loss / len(train_loader)
            print(f"\nEpoch [{epoch+1}/{epochs}] completed | Avg Loss: {avg_loss:.4f}")

            # ---------------- Validation ----------------
            model.classifier.eval()
            val_correct, val_total = 0, 0

            with torch.no_grad():
                val_bar = tqdm(val_loader, desc="Validating", leave=False)

                for audios, labels in val_bar:
                    labels = labels.to(device)

                    logits = model.forward(audios)
                    preds = logits.argmax(dim=1)

                    val_correct += (preds == labels).sum().item()
                    val_total += len(labels)

            val_acc = val_correct / val_total
            print(f"Validation accuracy: {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), VGGishFMA.MODEL_PATH)
                print(f"New best model saved (val_acc={val_acc:.4f})")

        print("Training complete.")

    # ----------------------------
    # Testing Loop
    # ----------------------------
    @staticmethod
    def test_generic(test_dataset, batch_size=2):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = VGGishFMA().to(device)

        # Load trained weights
        model.load_state_dict(torch.load(VGGishFMA.MODEL_PATH, map_location=device))
        model.classifier.eval()
        print("Loaded trained model.")

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=collate_fma
        )

        correct, total = 0, 0

        with torch.no_grad():
            test_bar = tqdm(test_loader, desc="Testing")

            for audios, labels in test_bar:
                labels = labels.to(device)

                logits = model.forward(audios)
                preds = logits.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += len(labels)

        print(f"Test accuracy: {correct / total:.4f}")