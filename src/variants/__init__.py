from src.variants.example import *
from src.variants.mel_cnn import *
from src.variants.mfcc_cnn import *
from src.variants.baseline import *
from src.variants.mel_cnn_v2 import *

__all__ = [
	'ExampleFMAModel',
	'MelCNNFMAModel',
    'MFCC_CNNFMAModel',
	'MelMLPFMAModel',
	'MelCNNFMAModelV2'
]
