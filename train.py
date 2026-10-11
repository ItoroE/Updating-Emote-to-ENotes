import torch
from torch import nn
from torchview import draw_graph
import torch.nn.functional as F


#cnn will act as the encoder for the vq-gan model
class CNNNet(nn.Module):
    def __init__(self):

        super().__init__()

        #1st block
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels =1,out_channels= 16, kernel_size =3, stride =1, padding =2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size =2)
        )
        #2nd block
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels =16,out_channels= 32, kernel_size =3, stride =1, padding =2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size =2)
        )

        #3rd block
        self.conv3 = nn.Sequential(nn.Conv2d( in_channels =32,out_channels= 64, kernel_size =3, stride =1, padding =2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size =2)
        )

        #4th block
        self.conv4 = nn.Sequential(nn.Conv2d( in_channels =64,out_channels= 128, kernel_size =3, stride =1, padding =2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size =2)
        )

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class CodebookQuant(nn.Module):
    # codebook size and embedding dim
    def __init__(self, num_embeddings, dim):
        super().__init__()
        #creating the shape of what codebook should look like
        self.codebook = nn.Parameter(torch.randn(num_embeddings, dim))

    def forward(self, x):
        B, C, H, W = x.shape
        # flattening feature maps
        residual = x.permute(0, 2, 3, 1).reshape(-1, C)
        z = residual
        output = torch.zeros_like(z)

        # choices of codebooks throught all 128 channels
        all_idx = []
        for c in self.codebook:
            idx = torch.cdist(residual,c)
            q = c[idx]
            #commitment loss
            residual = residual - q.detach()
            output += q
            all_idx.apppend(idx)
        codebook_loss = F.mse(output, z.detach())
        commit_loss = F.mse(z, output.detach())
        weighted_loss = codebook_loss + 0.25*commit_loss

        out = z + (out -z).detach()
        idx = torch.stack(all_idx, dim=-1).view(B,H,W,-1)
        return out, idx, weighted_loss


     


