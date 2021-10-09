from src.train import Trainer
from config import get_config
from logger.logger import Logger
from models.get_model import get_model
from dataloader.dataloader import get_dataloader


def main():
    config = get_config()
    model = get_model(
        model_name=config["model_name"],
        input_size=config["input_size"],
        num_classes=config["num_classes"]
    )
    dataloader = get_dataloader(config)
    logger = Logger
    trainer = Trainer(config=config,
                      model=model,
                      logger=logger,
                      dataloader=dataloader)
    model = trainer.run()


if __name__ == '__main__':
    main()
