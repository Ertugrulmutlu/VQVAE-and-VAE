from Utils.data_preprocessor import DataPreprocessor
from Utils.model import VQVAE
from Utils.train import VQTrainer

if __name__ == '__main__':
    preprocessor = DataPreprocessor(dataset_path=r".\\Datas", image_size=64, batch_size=64, shuffle=True)
    dataloader, channels = preprocessor.get_loader()

    model = VQVAE(in_channels=channels)
    trainer = VQTrainer(model=model, dataloader=dataloader)
    trainer.train(epochs=50, save_path="vqvae_model_128x128.pt")