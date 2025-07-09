import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMSpeakerEncoder(nn.Module) :
    def __init__(self, input_dim=40,hidden_dim=256,num_layer=3,embedding_dim=256):

        super(LSTMSpeakerEncoder,self).__init__()
        self.lstm=nn.LSTM(input_dim,hidden_dim,num_layer,batch_first=True)
        self.projection=nn.Linear(hidden_dim,embedding_dim)

    def forward(self,x):
        """
         x: Tensor of shape (batch, time, feature_dim)
        """
        print(f"tenssor shape-{x.shape}")
        lstm_out,_=self.lstm(x)
        last_frame=lstm_out[:,-1,:]
        embed=self.projection(last_frame) # (batch, embedding_dim)
        embed=F.normalize(embed,p=2,dim=1)
        return embed
