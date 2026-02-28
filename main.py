import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
import argparse

from torch.utils.data import DataLoader, random_split

from src.common import AbstractFMAGenreModule
from src.fma import VariableFMADataset, NoiseVariableFMADataset
from src.variants import *
from src.constants import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an FMA model')
    
    model_names = [cls.name() for cls in AbstractFMAGenreModule.__subclasses__()]
    parser.add_argument(
        'action', 
        choices=['train', 'test'], 
        help='The action to perform on the model'
    )
    parser.add_argument(
        'model',
        choices=model_names,
        help='The FMA model to run'
    )
    
    args = parser.parse_args()

    dataset = NoiseVariableFMADataset()

    n = len(dataset)

    test_sz = int(n * TEST_SPLIT)
    val_sz = int(n * VAL_SPLIT)
    train_sz = n - test_sz - val_sz

    train_ds, val_ds, test_ds = random_split(dataset, [train_sz, val_sz, test_sz])

    ModelClass = next(cls for cls in AbstractFMAGenreModule.__subclasses__() if cls.name() == args.model)

    if args.action == 'train':
        logging.info(f'[MAIN] Training model {ModelClass.name()} ({ModelClass.__name__})')
        ModelClass.train_generic(train_dataset=train_ds, val_dataset=val_ds)
    elif args.action == 'test':
        logging.info(f'[MAIN] Testing model {ModelClass.name()} ({ModelClass.__name__})')
        ModelClass.test_generic(test_dataset=test_ds)

    logging.info(f'[MAIN] Graceful end. Goodbye!')
