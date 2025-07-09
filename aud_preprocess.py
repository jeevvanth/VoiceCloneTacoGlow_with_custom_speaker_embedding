import torch
import numpy as np
import soundfile as sf
import torchaudio
import sys
sys.path.append('/home/jeevanth/Voice_cloning')
from tacotron2.model import Tacotron2
from tacotron2.text import text_to_sequence
from tacotron2.waveglow.glow import WaveGlow
from tacotron2.waveglow.denoiser import Denoiser
from tacotron2.hparams import create_hparams
from main import LSTMSpeakerEncoder
from huggingface_hub import hf_hub_download

hparams=create_hparams()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load models
tacotron2 = Tacotron2(hparams).to(device)
tacotron2.load_state_dict(torch.load("tacotron2_statedict.pt"))
tacotron2.eval()

waveglow = torch.load("waveglow_256channels_ljs_v3.pt")["model"].to(device).eval()
for m in waveglow.modules():
    if hasattr(m, "remove_weight_norm"):
        m.remove_weight_norm()
denoiser = Denoiser(waveglow)

speaker_encoder = LSTMSpeakerEncoder().to(device)
speaker_encoder.load_state_dict(torch.load("lstm_speaker_encoder.pt"))
speaker_encoder.eval()

def extract_mfcc(filepath, sample_rate=16000, n_mfcc=40):
    waveform, sr = torchaudio.load(filepath)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
    )
    mfcc = mfcc_transform(waveform).squeeze(0).transpose(0, 1)
    return mfcc

def encoder(audio_path):
    mfcc = extract_mfcc(audio_path).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = speaker_encoder(mfcc)
    return embedding  # (1, D)

def clean_audio():
    reference_audio = "clean.wav"
    text = "Hello, how are you?"

   
    speaker_embedding = encoder(reference_audio)
    text_sequence = text_to_sequence(text, ['english_cleaners'])
    text_sequence = torch.LongTensor(text_sequence).unsqueeze(0).to(device)
    mel_outputs, mel_postnet, _, _ = tacotron2.inference(text_sequence, speaker_embedding)
    mel_spectrogram = mel_postnet
    audio = waveglow.infer(mel_spectrogram, sigma=0.8)
    audio = denoiser(audio, 0.01).squeeze(1).cpu().numpy()

    sf.write("output.wav", audio[0], 22050)
    print("Output saved to output.wav")

if __name__ == "__main__":
    clean_audio()
