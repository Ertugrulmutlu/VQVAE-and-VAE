import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

# --- Ayarlar ---
image_path = "./Datas/test/test1.png"
reconstructed_path = "./Datas/test/reconstructed_50.jpg"  # JPEG kullandık
model_path = "vae_model_50.pt"

# --- Boyutunu ölç ---
original_size = os.path.getsize(image_path)

# --- Görseli yükle ve dönüştür ---
img = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
input_tensor = transform(img).unsqueeze(0)

# --- Modeli yükle ---
from Utils.model import VAE
model = VAE(latent_dim=128)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# --- Encode–Decode ---
with torch.no_grad():
    recon, _, _ = model(input_tensor)

# --- Yeniden görsele çevir ---
recon_img = recon.squeeze().permute(1, 2, 0).numpy()
recon_img = (recon_img * 255).astype(np.uint8)
recon_pil = Image.fromarray(recon_img)
recon_pil.save(reconstructed_path, quality=85)  # Düşük kalite JPEG

# --- Yeniden üretilmiş görselin boyutu ---
reconstructed_size = os.path.getsize(reconstructed_path)
compression_rate = original_size / reconstructed_size

# --- Görselleri göster ---
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img.resize((64, 64)))
axs[0].set_title(f"Original\n{original_size / 1024:.2f} KB")
axs[0].axis("off")

axs[1].imshow(recon_pil)
axs[1].set_title(
    f"Reconstructed\n{reconstructed_size / 1024:.2f} KB\n📉 {compression_rate:.2f}x smaller"
)
axs[1].axis("off")

plt.tight_layout()
plt.show()
