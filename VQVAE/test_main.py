import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

# --- Ayarlar ---
image_path = r".\\Datas\\test\\test1.png"
reconstructed_path = r".\\Datas\\test\\reconstructed_vqvae.jpg"
model_path = "vqvae_model.pt"

# --- Orijinal boyutu ölç ---
original_size = os.path.getsize(image_path)

# --- Görseli yükle ve dönüştür ---
img = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Eğitimde kullandığın boyut neyse onunla aynı olmalı
    transforms.ToTensor()
])
input_tensor = transform(img).unsqueeze(0)  # [1, 3, 64, 64]

# --- Modeli yükle ---
from Utils.model import VQVAE  # Modelinin dosya adı buysa
model = VQVAE()
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# --- Encode–Decode ---
with torch.no_grad():
    recon, _ = model(input_tensor)

# --- Yeniden görsele çevir ---
recon_img = recon.squeeze().permute(1, 2, 0).numpy()
recon_img = (recon_img * 255).astype(np.uint8)
recon_pil = Image.fromarray(recon_img)
recon_pil.save(reconstructed_path, quality=85)  # JPEG formatta kaydet

# --- Yeniden üretilmiş görselin boyutu ---
reconstructed_size = os.path.getsize(reconstructed_path)

# --- Sonuçları yazdır ---
compression_rate = original_size / reconstructed_size
print(f"🖼️ Original:      {original_size / 1024:.2f} KB")
print(f"🔁 Reconstructed: {reconstructed_size / 1024:.2f} KB")
print(f"📉 Compression Rate: {compression_rate:.2f}x")

# --- Görselleri göster ---
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(img)
axs[0].set_title("Original")
axs[0].axis("off")
axs[1].imshow(recon_pil)
axs[1].set_title("Reconstructed (VQ-VAE)")
axs[1].axis("off")
plt.tight_layout()
plt.show()

