import torchaudio
import torch
import librosa
from main import LSTMSpeakerEncoder

def extract_features(waveform,sample_rate,n_mfcc=40):
    """
    """
    waveform = torch.tensor(waveform).unsqueeze(0)


    mfcc_transform=torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft":400,"hop_length":160,"n_mels":40}
    )
    mfcc = mfcc_transform(waveform)  # (1, 40, time)
    mfcc = mfcc.transpose(1, 2)

    return mfcc






encoder = LSTMSpeakerEncoder()
# sample_audio, sr = torchaudio.load("clean.wav")
sample_audio, sr = librosa.load("Human Voice1.wav", sr=16000)
# waveform = torch.tensor(sample_audio).unsqueeze(0)
features = extract_features(sample_audio, sr)  # shape: (1, time, 40)
embedding = encoder(features)  # shape: (1, embedding_dim)

print(embedding)