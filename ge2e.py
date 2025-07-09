import torch
import torch.nn as nn
import torch.nn.functional as F

class GE2ELoss(nn.Module):
    def __init__(self,init_w=10.0,init_b=-5.0):
        super(GE2ELoss,self).__init__()
        self.w=nn.Parameter(torch.tensor(init_w))
        self.b=nn.Parameter(torch.tensor(init_b))

    def forward(self,embeddings):
        """
        """

        N,M,D=embeddings.size()
        centroids=torch.mean(embeddings,dim=1)

        sim_matrix=torch.zeros(N,M,N).to(embeddings.device)
        for j in range(N):
            for m in range(M):
                for k in range(N):
                    if j == k:
                        excl = torch.cat([embeddings[j, :m], embeddings[j, m+1:]], dim=0)
                        c = torch.mean(excl, dim=0)
                    else: 
                         c = centroids[k]
                    sim_matrix[j, m, k] = F.cosine_similarity(embeddings[j, m], c, dim=0)
        sim_matrix = self.w * sim_matrix + self.b

        labels = torch.arange(N).unsqueeze(1).expand(N, M).reshape(-1).to(embeddings.device)
        loss = F.cross_entropy(sim_matrix.reshape(N*M, N), labels)

        return loss