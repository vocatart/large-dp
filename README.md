# large-dp

large-dp is a collection of helper scripts for [DeepPhonemizer](https://github.com/as-ideas/DeepPhonemizer).
It is designed to aid in training g2p models that have many languages, graphemes, and phonemes.

## installation

Before cloning this repo, make sure you have deep-phonemizer and a relevant [pytorch](https://pytorch.org/get-started/locally/) distribution installed, preferably in a separate python environment.

## setup

large-dp only requires two configuration files, and a directory containing dictionaries.

### dictionaries

Dictionaries should be organized as follows:

```text
language1
    dictionary1.tsv
    dictionary2.tsv
language2
    dictionary1.tsv
    dictionary2.tsv
etc...
```

### configurations

large-dp requires two configuration files, one containing linguistic information, and one containing all the DeepPhonemizer arguments.
Example configs are available under `configs`.

#### example_linguistic.yaml

The linguistics configuration only needs two fields, a path to the dictionary directory and a list of all language and their dictionaries.
The language name is used as the first entry token in DeepPhonemizer.

```yaml
linguistic_path: ../linguistics
languages:
  - name: English
    dictionaries:
      - Dictionary1
      - Dictionary2
# etc
```

#### example_config.yaml

The master configuration should contain all fields needed for DeepPhonemizer, without the `text_symbols` and `phoneme_symbols` under the `preprocessing` section.
These are generated during preprocessing. Additionally, this file should contain a new `validation` field that looks as follows:

```yaml
# settings for obtaining validation data for each language
validation:
  validation_percentage: 0.01
  validation_minimum: 5
  validation_maximum: 100
```

## usage

### preprocessing

The preprocessing script uses the following format. `exp_name` should be consistent across all scripts, as it determines logging directory and artifact names.
This helper script will read through all dictionaries defined in the linguistic configuration, and generate a dataset and configuration file that contains all necessary graphemes, phonemes, and words.
Binarized DeepPhonemizer datasets are saved under `experiments/exp_name`.

```text
usage: preprocess.py [-h] config linguistic exp_name

positional arguments:
  config      Config file for DP training
  linguistic  Linguistic file for DP preprocessing
  exp_name    Experiment name
```

### training

The training script uses the following format. It will not resume from checkpoint by default unless given a checkpoint path with the `-c` flag.
Training logs are saved under `experiments/exp_name`.

```text
usage: train.py [-h] [-c CHECKPOINT] exp_name

positional arguments:
  exp_name              Experiment name

options:
  -c, --checkpoint CHECKPOINT
                        Checkpoint file to resume from
```

### exporting

Models can be traced and exported via TorchScript. `best_model.pt` under `experiments/exp_name` is used for exporting. Traced models are saved under `artifacts/exp_name.pt`

```text
usage: export.py [-h] exp_name

positional arguments:
  exp_name    Experiment name
```
