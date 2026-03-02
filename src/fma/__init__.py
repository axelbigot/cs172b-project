from src.fma.fma_dataset import *
from src.fma.fma_utils import *
from src.fma.dataset_analyzer import *
from src.fma.datased_fused_dataset import *
from src.fma.mel_dataset import *
from src.fma.mfcc_dataset import *


class DatasedFusedDataset(DatasedFusedMixin, VariableFMADataset):
	pass
class MelFMADataset(MelPrecomputeMixin, VariableFMADataset):
	pass
class MelNoiseDataset(MelPrecomputeMixin, DatasedFusedDataset):
	pass
class MfccFMADataset(MfccPrecomputeMixin, VariableFMADataset):
	pass
class MfccNoiseDataset(MfccPrecomputeMixin, DatasedFusedDataset):
	pass
class MelNoiseMaskingAugmentDataset(MelAugmentMixin, MelNoiseDataset):
	pass

__all__ = [
	'VariableFMADataset',
	'compare_splits',
	'DatasedFusedDataset',
	'MelFMADataset',
	'MelNoiseDataset',
	'MfccFMADataset',
	'MfccNoiseDataset',
	'MelNoiseMaskingAugmentDataset'
]
