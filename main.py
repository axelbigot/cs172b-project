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
    'fma+noise+mel+mask': MelNoiseMaskingAugmentDataset
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an FMA model')
    subparsers = parser.add_subparsers(dest='action', required=True)
    
    model_names = [cls.name() for cls in AbstractFMAGenreModule.__subclasses__()]

    custom_parser = subparsers.add_parser('custom')

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

    for p in [train_parser, test_parser, custom_parser]:
        p.add_argument(
            '--tag',
            type=str,
            default='',
            help='Specify a tag to append to the name of the model to make it unique'
        )

    custom_parser.add_argument(
        '--frac',
        type=float,
        default=1.0,
        help='Fraction of dataset to use [0,1]'
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

    if args.action == 'custom':
        tst_accuracies = {}

        tr_dataset = MelNoiseMaskingAugmentDataset(split='training', downsample_frac=args.frac)
        val_dataset = MelNoiseMaskingAugmentDataset(split='validation', downsample_frac=args.frac, genre_encoder=tr_dataset.genre_encoder)
        tst_dataset = MelNoiseMaskingAugmentDataset(split='test', downsample_frac=args.frac, genre_encoder=tr_dataset.genre_encoder)

        for lr in [1e-3, 5e-4, 1e-4]:
            for temporal_kernel in [3, 5]:
                for batch_size in [8, 16, 32]:
                    unique_tag = f'tg-{args.tag}_lr-{lr}_tempk-{temporal_kernel}_bs-{batch_size}'
                    model = MelCNNFMAModelV2(
                        tr_dataset.num_classes,
                        tag=unique_tag,
                        conv_channels=(64,64,128),
                        temporal_kernel_size=temporal_kernel
                    )
                    model.fma_train(
                        tr_dataset,
                        val_dataset,
                        batch_size=batch_size,
                        optimizer=None,
                        criterion=None,
                        lr=lr,
                        num_epochs=150,
                        early_stopping_patience=10
                    )
                    tst_acc = model.fma_test(tst_dataset)
                    tst_accuracies[unique_tag] = tst_acc
                    print(f'{unique_tag} : {tst_acc*100:.2f}%')

        for tag in sorted(tst_accuracies, key=tst_accuracies.get, reverse=True):
            print(f'{tag} : {tst_accuracies[tag]*100:.2f}%')

    else:
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

                ModelClass.train_generic(train_dataset=train_ds, val_dataset=val_ds, tag=args.tag)
            elif args.action == 'test':
                logging.info(f'[MAIN] Testing model {ModelClass.name()} ({ModelClass.__name__})')
                test_ds = DatasetClass(split='test', downsample_frac=frac, genre_encoder=train_ds.genre_encoder)

                ModelClass.test_generic(test_dataset=test_ds, tag=args.tag)


    logging.info(f'[MAIN] Graceful end. Goodbye!')
