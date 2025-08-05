import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from Utils.data_preprocessor import DataPreprocessor
from Utils.model import VAE  # make sure this matches your filename

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 dataloader: DataLoader,
                 device: torch.device = None,
                 lr: float = 1e-3):
        """
        Trainer class for training a VAE model.

        Args:
            model (torch.nn.Module): Your VAE model.
            dataloader (DataLoader): PyTorch dataloader with input images.
            device (torch.device): 'cuda' or 'cpu'.
            lr (float): Learning rate.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def loss_function(self, recon_x, x, mu, logvar):
        """
        Computes VAE loss = reconstruction loss + KL divergence.
        """
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div

    def train(self, epochs: int = 20, save_path: str = "vae_model.pt"):
        """
        Trains the VAE model.

        Args:
            epochs (int): Number of training epochs.
            save_path (str): Path to save the trained model.
        """
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch, _ in self.dataloader:
                batch = batch.to(self.device)

                self.optimizer.zero_grad()
                x_hat, mu, logvar = self.model(batch)
                loss = self.loss_function(x_hat, batch, mu, logvar)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader.dataset)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        # Save the trained model
        torch.save(self.model.state_dict(), save_path)
        print(f"✅ Model saved to: {save_path}")
