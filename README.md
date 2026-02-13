# cs172b-project

Comparison of various approaches to genre classification of variable-length audio sequences (< 30s)

## Prerequisites

From [FMA](https://github.com/mdeff/fma/tree/master):
- `fma_metadata.zip` in the project root directory
- `fma_small.zip` in the project root directory
- `pip install -r requirements.txt` (run in a venv)

## Running

`python main.py -h`

Then run using a specific model, for instance the demo:

`python main.py example`

You can view tensorboard visuals with:

`tensorboard --logdir data/runs/`

With `model_name` being the provided name of the model (`example`)

## Unified Process

The main program will do the following:
1. Dynamically retrieve the model class to be run based on the `model` arg
2. Invoke the model's static `train_generic` method, which should delegate to the model's `fma_train` method.
3. The `fma_train` method will instantiate an instance of the dataset (`VariableFMADataset`) and create a data loader for it.
4. The dataset will unzip `fma_metadata.zip` and `fma_small.zip` to `data/` if not already extracted
5. The dataset will load metadata CSVs into memory using `pandas` and restrict to the `small` subset of FMA
6. Common training loop in `fma_train` will begin. The dataloader will retrieve `FMATrack` instances of `batch_size` and collate them to `FMATrackBatch`, which is what is provided to models' `forward()` method. This object contains an aggregate of `FMATrack` scalars as tensors, as well as a raw array of audio bytes which certain models may need to preprocess (i.e. pad) to produce torch-compatible tensors. When the dataloader internally calls the dataset's `__getitem__`, audio files will be trimmed to a uniformly random sequence of 10-30s (at any point in the audio).

## Adding a Model

1. Create a file or module (if multiple files) in `src/variants`
2. Subclass `AbstractFMAGenreModule` and implement the abstract methods `forward`, `train_generic`, and `name`. Note that `train_generic` should create an instance of the model and call its `fma_train` method (see `src/variants/example.py` for example).
3. in `src/variants/__init__.py` export the new model class.

The model should automatically have a `model` argument option in the main program CLI with the value of `name` and can be ran just like the example.

> Note: the `RANDOM_SEED` variable in `src/constants.py` should be used anywhere where random state is configured (torch, rand, etc...) for reproducibility reasons.

## Unified TODO:

- Will add model saving to disk so that trained models persist
- Will add train, test, validation data splits
- Will add args to train vs test vs validate