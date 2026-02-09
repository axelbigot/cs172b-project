import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
import argparse

from torch.utils.data import DataLoader
from src.common import AbstractFMAGenreModule
from src.variants import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an FMA model')
    
    model_names = [cls.name() for cls in AbstractFMAGenreModule.__subclasses__()]
    parser.add_argument(
        'model',
        choices=model_names,
        help='The FMA model to run'
    )
    
    args = parser.parse_args()

    ModelClass = next(cls for cls in AbstractFMAGenreModule.__subclasses__() if cls.name() == args.model)
    logging.info(f'[MAIN] Training model {ModelClass.name()} ({ModelClass.__name__})')
    ModelClass.train_generic()

    logging.info(f'[MAIN] Graceful end. Goodbye!')
