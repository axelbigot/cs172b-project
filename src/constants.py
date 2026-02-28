from pathlib import Path


# Global seed for all random states. This value should be used
# anywhere where a random seed can be passed.
RANDOM_SEED = 1

############################### MODEL ###############################

# Number of genres (output space)
NUM_CLASSES = 9
# Validation frac split
VAL_SPLIT = 0.1
# Test frac split
TEST_SPLIT = 0.2

############################## DATASET ##############################

# Data directory
DATA_DIRECTORY = Path('data')
# Metadata zip source file
METADATA_ZIP = Path('fma_metadata.zip')
