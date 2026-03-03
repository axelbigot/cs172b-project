import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
import argparse

from torch.utils.data import DataLoader, random_split

from src.common import AbstractFMAGenreModule
from src.fma import VariableFMADataset
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
    parser.add_argument(
        '--frac', 
        type=float, 
        default=1.0,
        help='Fraction of the dataset to use for training/testing (0-1)'
    )
    
    args = parser.parse_args()

    dataset = VariableFMADataset()
    n = len(dataset)

    # Apply fraction safely using random_split
    if args.frac < 1.0:
        n_frac = int(n * args.frac)
        logging.info(f'[MAIN] Using {n_frac}/{n} samples ({args.frac*100:.0f}%) for this run.')
        dataset, _ = random_split(dataset, [n_frac, n - n_frac])
        n = len(dataset)  # update n

    test_sz = int(n * TEST_SPLIT)
    val_sz = int(n * VAL_SPLIT)
    train_sz = n - test_sz - val_sz

    train_ds, val_ds, test_ds = random_split(dataset, [train_sz, val_sz, test_sz])

    ModelClass = next(cls for cls in AbstractFMAGenreModule.__subclasses__() if cls.name() == args.model)

    if args.action == 'train':
        model = ModelClass()
        logging.info(f'[MAIN] Training model {model.name()} ({ModelClass.__name__})')
        model.train_generic(train_dataset=train_ds, val_dataset=val_ds)

    elif args.action == 'test':
        model = ModelClass()
        logging.info(f'[MAIN] Testing model {model.name()} ({ModelClass.__name__})')
        model.test_generic(test_dataset=test_ds)

    logging.info(f'[MAIN] Graceful end. Goodbye!')