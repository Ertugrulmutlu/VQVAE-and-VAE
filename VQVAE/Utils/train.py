import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class VQTrainer:
    def __init__(self, model, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.recon_loss_fn = nn.MSELoss()

    def train(self, epochs=50, save_path="vqvae_model.pt"):
        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0
            total_vq_loss = 0
            total_recon_loss = 0

            for batch, _ in tqdm(self.dataloader, desc=f"Epoch {epoch}/{epochs}"):
                batch = batch.to(self.device)
                recon, vq_loss = self.model(batch)

                recon_loss = self.recon_loss_fn(recon, batch)
                loss = recon_loss + vq_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_vq_loss += vq_loss.item()
                total_recon_loss += recon_loss.item()

            print(f"Epoch {epoch}/{epochs} | Total Loss: {total_loss:.4f} | Recon Loss: {total_recon_loss:.4f} | VQ Loss: {total_vq_loss:.4f}")

        torch.save(self.model.state_dict(), save_path)
