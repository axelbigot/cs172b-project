import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
import argparse

from torch.utils.data import DataLoader, random_split

from src.common import AbstractFMAGenreModule
from src.fma import *
from src.variants import *
from src.constants import *


DATASET_MAP: dict[str, type[VariableFMADataset]] = {
    'fma': VariableFMADataset,
    'fma+noise': DatasedFusedDataset,
    'fma+mel': MelFMADataset,
    'fma+mfcc': MfccFMADataset,
    'fma+noise+mel': MelNoiseDataset,
    'fma+noise+mfcc': MfccNoiseDataset,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an FMA model')
    subparsers = parser.add_subparsers(dest='action', required=True)
    
    model_names = [cls.name() for cls in AbstractFMAGenreModule.__subclasses__()]

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument(
        'model',
        choices=model_names,
        help='The FMA model to train'
    )

    test_parser = subparsers.add_parser('test')
    test_parser.add_argument(
        'model',
        choices=model_names,
        help='The FMA model to test'
    )

    analyze_parser = subparsers.add_parser('analyze-ds')

    for p in [train_parser, test_parser, analyze_parser]:
        p.add_argument(
            '--frac',
            type=float,
            default=1.0,
            help='Fraction of dataset to use [0,1]'
        )

        p.add_argument(
            'dataset',
            choices=DATASET_MAP.keys(),
            help='Dataset to use'
        )
    
    args = parser.parse_args()

    frac = args.frac
    DatasetClass: type[VariableFMADataset] = DATASET_MAP[args.dataset]

    if args.action == 'analyze-ds':
        logging.info(f'[MAIN] (Re)Generating dataset analysis')
        train_ds = DatasetClass(split='training', downsample_frac=frac)
        val_ds = DatasetClass(split='validation', downsample_frac=frac, genre_encoder=train_ds.genre_encoder)
        test_ds = DatasetClass(split='test', downsample_frac=frac, genre_encoder=train_ds.genre_encoder)

        trn_an = train_ds.analyzer()
        val_an = val_ds.analyzer()
        tst_an = test_ds.analyzer()

        for an in [trn_an, val_an, tst_an]:
            an.simple()
            an.visual()

        compare_splits(trn_an, val_an, tst_an)
    else:
        ModelClass = next(cls for cls in AbstractFMAGenreModule.__subclasses__() if cls.name() == args.model)
        train_ds = DatasetClass(split='training', downsample_frac=frac)

        if args.action == 'train':
            logging.info(f'[MAIN] Training model {ModelClass.name()} ({ModelClass.__name__})')
            val_ds = DatasetClass(split='validation', downsample_frac=frac, genre_encoder=train_ds.genre_encoder)

            ModelClass.train_generic(train_dataset=train_ds, val_dataset=val_ds)
        elif args.action == 'test':
            logging.info(f'[MAIN] Testing model {ModelClass.name()} ({ModelClass.__name__})')
            test_ds = DatasetClass(split='test', downsample_frac=frac, genre_encoder=train_ds.genre_encoder)

            ModelClass.test_generic(test_dataset=test_ds)


    logging.info(f'[MAIN] Graceful end. Goodbye!')
