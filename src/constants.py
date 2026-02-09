from pathlib import Path


# Global seed for all random states. This value should be used
# anywhere where a random seed can be passed.
RANDOM_SEED = 1

############################### MODEL ###############################

# Number of genres (output space)
NUM_CLASSES = 8

############################## DATASET ##############################

# Data directory
DATA_DIRECTORY = Path('data')
# Metadata zip source file
METADATA_ZIP = Path('fma_metadata.zip')
