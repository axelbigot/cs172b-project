import torch
import torch.nn as nn
import torchaudio

from src.common import AbstractFMAGenreModule

class CRNNGenreModel(AbstractFMAGenreModule):

    @classmethod
    def name(cls):
        return "crnn"
    
    @classmethod
    def train_generic(cls, train_dataset, val_dataset, tag=None, **kwargs):
        model = cls(tag)
        model.fma_train(train_dataset, val_dataset, batch_size=32, num_epochs=30)

    @classmethod
    def test_generic(cls, test_dataset, tag=None, **kwargs):
        model = cls(tag)
        test_accuracy = model.fma_test(test_dataset)
        print(f"\n[INFO] CRNN Test Accuracy: {(test_accuracy * 100):.2f}%")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sample_rate = 22050

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # 4 CNN Layers
        # 1. Edges
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 2. Tempos
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 3. Chords
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 4. Genre indicators (complex shapes)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.cnn_output_height = 8
        self.cnn_output_channels = 256

        self.rnn_input_size = self.cnn_output_channels * self.cnn_output_height

        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(128 * 2, 9)

    def forward(self, batch_X, ids):
        device = next(self.parameters()).device
        audios = [load_audio(device) for load_audio in batch_X]
        padded_audio = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
        x = padded_audio.unsqueeze(1)
        x = x.to(device)

        x = self.melspec(x)
        x = self.amplitude_to_db(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.permute(0, 3, 1, 2)
        batch_size, time_steps, c, f = x.size()
        x = x.reshape(batch_size, time_steps, c * f)
        x, _ = self.rnn(x)

        x, _ = torch.max(x, dim=1)      # Keeps the max timestep for each feature
        logits = self.fc(x)

        return logits