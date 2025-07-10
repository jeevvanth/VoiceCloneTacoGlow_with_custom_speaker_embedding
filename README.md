# VoiceCloneTacoGlow_with_custom_speaker_embedding

A clean, modular implementation of voice cloning using a custom speaker encoder, Tacotron2 for text-to-spectrogram, and WaveGlow for spectrogram-to-waveform vocoding. This project allows you to generate speech in the voice of any speaker given just one reference audio file.

# Objective

Clone a speaker's voice from a reference audio and arbitrary text input using:

* LSTM-based Speaker Encoder trained with GE2E loss

* Tacotron2 for text-to-mel spectrogram conversion

* WaveGlow vocoder for mel-to-waveform synthesis

# Setup

Need to have two virtual environment for separating tensorflow and pytorch. Most of our code is developed by the pytorch and you have to install the tensorflow==2.13.1 typing-extensions==4.5.0 where problem is typing-extension 

```bash
python3 -m venv venv_torch
source venv_torch/bin/activate

pip install -r requirements.txt
```
# For Tensor flow
```bash
python3 -m venv venv_tf
source venv_tf/bin/activate

pip install tensorflow==2.13.1 typing-extensions==4.5.0
```

# Installation for tacotron and Waveglow
```bash
pip install git+https://github.com/NVIDIA/tacotron2.git
pip install git+https://github.com/NVIDIA/waveglow.git
```
Download pretrained weights:

[Tacotron2](https://colab.research.google.com/drive/1VAuIqEAnrmCig3Edt5zFgQdckY9TDi3N#scrollTo=MN72YKKvb8pM)

[WaveGlow](https://colab.research.google.com/drive/1VAuIqEAnrmCig3Edt5zFgQdckY9TDi3N#scrollTo=MN72YKKvb8pM)

# 📜 License

This project is licensed under the MIT License.

Tacotron2 and WaveGlow are originally developed by NVIDIA and licensed under the BSD 3-Clause License.

[Tacotron2 GitHub](https://github.com/NVIDIA/tacotron2)

[WaveGlow GitHub](https://github.com/NVIDIA/tacotron2)

Pretrained checkpoints and core TTS modules are used in compliance with their respective licenses.
