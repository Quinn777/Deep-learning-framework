from src.train import Trainer
from src.utils.config import get_config


def main():
    config = get_config()
    trainer = Trainer(config)
    model = trainer.run()


if __name__ == '__main__':
    main()
