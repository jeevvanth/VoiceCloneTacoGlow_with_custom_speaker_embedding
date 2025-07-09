import os
import glob
import torch
import torchaudio
import random
from torch.utils.data import Dataset, DataLoader
from main import LSTMSpeakerEncoder
from ge2e import GE2ELoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpeakerDataset(Dataset):
    def __init__(self,root_dir,n_mfcc=40,sample_rate=16000,M=5,max_len = 300):
        self.sample_rate=sample_rate
        self.M=M
        self.n_mfcc=n_mfcc
        self.max_len = max_len
        self.speakers={}
        for filepath in glob.glob(os.path.join(root_dir,"*")):
            speaker=os.path.basename(filepath).split('_')[0]
            self.speakers.setdefault(speaker, []).append(filepath)

        print(f"Total speakers before filtering: {len(self.speakers)}")
        self.speakers={k: v for k,v in self.speakers.items() if len(v)>=M}
        self.speaker_list= list(self.speakers.items())
        print(f"Total speakers after filtering for M={self.M}: {len(self.speakers)}")


    def __len__(self):
        return len(self.speaker_list)
    
    def __getitem__(self, idx):
        speaker, files = self.speaker_list[idx]
        selected_files = random.sample(files, self.M)
        mfccs = [self._extract_mfcc(f) for f in selected_files]
        mfccs = [self._pad_or_truncate(m) for m in mfccs]
        return torch.stack(mfccs)
    
    def _pad_or_truncate(self, mfcc):
        """
        Pad or truncate mfcc to (max_len, n_mfcc)
        """
        T, D = mfcc.shape
        if T > self.max_len:
            return mfcc[:self.max_len, :]
        else:
            padding = torch.zeros(self.max_len - T, D)
            return torch.cat([mfcc, padding], dim=0)

    def _extract_mfcc(self, filepath):
        waveform, sr = torchaudio.load(filepath)
        if sr != self.sample_rate:
            resample = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resample(waveform)
        mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
        )
        return mfcc(waveform).squeeze(0).transpose(0, 1)  # (time, n_mfcc)

# Collate N speakers × M utterances
def collate_fn(batch):
    return torch.stack(batch)  # (N, M, time, n_mfcc)

def train():
    N,M=4,5
    dataset=SpeakerDataset("data",M=M)
    dataloader=DataLoader(dataset,batch_size=N,shuffle=True)


    encoder=LSTMSpeakerEncoder().to(device)
    criterion=GE2ELoss().to(device)
    optimizer=torch.optim.Adam(list(encoder.parameters()) + list(criterion.parameters()), lr=1e-4)

    for epoch in range(10):
        for batch in dataloader:
            batch=batch.to(device) # (N, M, T, D)
            N,M,T,D=batch.shape
            batch=batch.view(N*M,T,D)
            embeddings=encoder(batch)
            embeddings=embeddings.view(N, M, -1)  # (N, M, D)

            loss = criterion(embeddings) #loss func
            optimizer.zero_grad() #gradient func
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 3.0)
            optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    torch.save(encoder.state_dict(), "lstm_speaker_encoder.pt")
    print("Saved speaker encoder as lstm_speaker_encoder.pt")


if __name__=="__main__":
    train()




        