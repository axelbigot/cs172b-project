import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from torchvggish import vggish
from tqdm import tqdm
from src.common import AbstractFMAGenreModule

# ----------------------------
# Collate function for FMA
# ----------------------------
def collate_fma(batch, device='cpu', precomputed_mels=False):
    """
    Collate function for FMA datasets.
    Supports:
      - raw waveform: (audio_loader, label, idx)
      - precomputed mels: (mel_tensor, label, idx)
    """
    if precomputed_mels:
        mels = torch.stack([t[0] for t in batch]).to(device)
        labels = torch.tensor([t[1] for t in batch], dtype=torch.long).to(device)
        ids = [t[2] for t in batch]
        return mels, labels, ids
    else:
        audios = [t[0] for t in batch]  # lazy loaders
        labels = torch.tensor([t[1] for t in batch], dtype=torch.long)
        ids = [t[2] for t in batch]
        return audios, labels, ids


# ----------------------------
# VGGish model for FMA
# ----------------------------
class VGGishFMA(AbstractFMAGenreModule):
    MODEL_PATH = "vggish_fma_best.pt"

    def __init__(self, tag: str = "", use_precomputed_mels=False):
        super().__init__(tag)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_precomputed_mels = use_precomputed_mels

        # VGGish backbone
        self.model = vggish().to(self.device)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 16)
        ).to(self.device)

        # Audio transforms (only needed if raw waveform)
        self.target_sr = 16000
        self.resampler = torchaudio.transforms.Resample(22050, self.target_sr)
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=400,
            hop_length=160,
            n_mels=64
        )

    # ----------------------------
    # Abstract method implementations
    # ----------------------------
    @classmethod
    def name(cls):
        return "vggish"

    def train_generic(self, train_dataset, val_dataset, **kwargs):
        return self.fma_train(train_dataset, val_dataset, **kwargs)

    def test_generic(self, test_dataset, **kwargs):
        return self.fma_test(test_dataset, **kwargs)

    @classmethod
    def collate_fn(cls):
        return lambda batch: collate_fma(batch, device='cpu', precomputed_mels=getattr(cls, 'use_precomputed_mels', False))

    # ----------------------------
    # Forward pass
    # ----------------------------
    def forward(self, batch, ids=None):
        if self.use_precomputed_mels:
            embeddings = self.model(batch)
            logits = self.classifier(embeddings)
            return logits
        else:
            all_examples, batch_sizes = self.waveform_to_examples(batch)
            embeddings = self.model(all_examples)

            # average embeddings per track
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
    # Convert waveform to VGGish examples
    # ----------------------------
    def waveform_to_examples(self, waveforms: list, srs: list = None):
        if srs is None:
            srs = [22050] * len(waveforms)

        batch_examples = []
        batch_sizes = []
        frame_size = 96

        for waveform, sr in zip(waveforms, srs):
            # Call the loader if it's a callable, pass device
            if callable(waveform):
                waveform = waveform(self.device)

            # Ensure tensor type
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform)

            if sr != self.target_sr:
                waveform = self.resampler(waveform.unsqueeze(0)).squeeze(0)

            min_len = self.mel_spec.n_fft + (frame_size - 1) * self.mel_spec.hop_length
            if waveform.numel() < min_len:
                pad_size = min_len - waveform.numel()
                waveform = torch.nn.functional.pad(waveform, (0, pad_size))

            mel = self.mel_spec(waveform.unsqueeze(0))
            log_mel = torch.log(mel + 1e-6).squeeze(0).transpose(0, 1)

            # Pad frames if needed
            if log_mel.size(0) < frame_size:
                pad_frames = frame_size - log_mel.size(0)
                log_mel = torch.cat([log_mel, torch.zeros(pad_frames, log_mel.size(1), device=log_mel.device)], dim=0)

            num_examples = log_mel.size(0) // frame_size
            examples = log_mel[:num_examples * frame_size].unfold(0, frame_size, frame_size)
            if examples.dim() == 2:
                examples = examples.unsqueeze(0)
            examples = examples.unsqueeze(1)

            batch_examples.append(examples)
            batch_sizes.append(examples.size(0))

        all_examples = torch.cat(batch_examples, dim=0).to(self.device)
        return all_examples, batch_sizes
    # ----------------------------
    # Transform batch for precomputed mels
    # ----------------------------
    def transform_batch(self, dataset, x, ids):
        if self.use_precomputed_mels:
            return x.to(self.device)
        return x

    # ----------------------------
    # Override get_idstr to handle Subset datasets
    # ----------------------------
    def get_idstr(self, dataset):
        base_dataset = getattr(dataset, 'dataset', dataset)  # unwrap Subset if needed
        frac = getattr(base_dataset, 'dowsample_frac', 'NA')
        return f'model_trained_{self.name()}{f"_tag-{self.tag}" if len(self.tag) else ""}_{base_dataset.__class__.__name__}_frac-{frac}'