def main():
    from Utils.data_inspector import DataInspector
    from Utils.data_preprocessor import DataPreprocessor
    from Utils.train import Trainer
    from Utils.model import VAE

    preprocessor = DataPreprocessor(
        dataset_path=r".\\Datas",
        image_size=64,
        batch_size=64,
        shuffle=True
    )
    dataloader, channels = preprocessor.get_loader()

    model = VAE(latent_dim=128)

    trainer = Trainer(model=model, dataloader=dataloader)
    trainer.train(epochs=50, save_path="vae_model_50.pt")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
