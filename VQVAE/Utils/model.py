import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        # z: [B, C, H, W] -> [BHW, C]
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z_perm.view(-1, self.embedding_dim)

        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_z, self.embeddings.weight.t())
            + torch.sum(self.embeddings.weight ** 2, dim=1)
        )

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices).view(z_perm.shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        loss = F.mse_loss(quantized.detach(), z) + self.commitment_cost * F.mse_loss(quantized, z.detach())
        quantized = z + (quantized - z).detach()

        return quantized, loss

class VQVAE(nn.Module):
    def __init__(self, in_channels=3, embedding_dim=64, num_embeddings=512, commitment_cost=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),           # 16x16
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, 3, 1, 1) # 16x16
        )

        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 64, 4, 2, 1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),             # 64x64
            nn.ReLU(),
            nn.Conv2d(32, in_channels, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss = self.vq(z)
        recon = self.decoder(quantized)
        return recon, vq_loss
