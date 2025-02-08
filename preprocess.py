import argparse
import csv
import os.path
import random
import dp.preprocess
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import SingleQuotedScalarString
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    """
    Reads .yaml configuration file into dictionary.

    :param config_path: Path to .yaml file.
    :return: YAML dictionary.
    """
    yaml_loader = YAML(typ='rt')
    yaml_loader.default_flow_style = None
    yaml_loader.preserve_quotes = True

    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml_loader.load(file)


def read_dictionary(dict_path: str, ling_path: str, lang_name) -> list[tuple[str, list[str]]]:
    """
    Reads target .tsv file into list of word entries with language and phoneme info.

    :param dict_path: Path to .tsv file.
    :param ling_path: Path to linguistic .yaml configuration file.
    :param lang_name: Language name to add to each entry.
    :return: List of word entries with language name, word, and list of phonemes.
    """
    dict_data = []

    with open(os.path.join(os.path.abspath(ling_path), lang_name, dict_path + ".tsv"), 'r', encoding='utf-8') as dict_file:
        reader = csv.reader(dict_file, delimiter='\t')

        for row in reader:
            if len(row) == 2:
                dict_data.append((row[0], row[1].split()))
            else:
                print(f'Malformed dictionary at {dict_path}')
                exit(1)

    return dict_data


def process_languages(ling_data: dict) -> list[list[list[tuple[str, str, list[str]]] | str]]:
    """
    Takes loaded linguistic configuration and returns all languages inside as a list of each language name and all of its entries.

    :param ling_data: Linguistic YAML dictionary.
    :return: List of all languages and their entries. Each entry contains the language name, word, and list of phonemes.
    """

    ling_path = ling_data.get("linguistic_path")
    retrieved_languages = ling_data.get("languages", [])
    data = []

    with tqdm(total=len(retrieved_languages)) as bar:
        for found_langauge in retrieved_languages:
            name = str(found_langauge["name"])
            all_words = []
            bar.set_description(f"Loading {name}")

            for dictionary in found_langauge.get("dictionaries", []):
                all_words.extend(read_dictionary(dictionary, ling_path, name))

            data_entries = []
            for dict_entry in all_words:
                data_entries.append((name, dict_entry[0], dict_entry[1]))

            data.append([data_entries, name])

            bar.update(1)

    bar.close()

    return data


def collate_symbols(language_data: list[list[list[tuple[str, str, list[str]]] | str]]) -> tuple[set[str], set[str]]:
    """
    Reads through all languages and returns the unique graphemes and phonemes, combined.

    :param language_data: Language dictionary object.
    :return: Tuple contaning a set of all graphemes, and a set of all phonemes.
    """
    grapheme_set = set()
    phoneme_set = set()

    for language in language_data:
        for entry in language[0]:
            graphemes_split = [grapheme for grapheme in str(entry[1])]
            grapheme_set.update(graphemes_split)
            phoneme_set.update(entry[2])

    return grapheme_set, phoneme_set


def create_dp_config(master_config: dict, loaded_graphemes: set, loaded_phonemes: set, all_langs: list[str], exp: str) -> dict:
    """
    Creates DeepPhonemizer configuration dictionary for given master configuration.

    :param master_config: Loaded configuration dictionary.
    :param loaded_graphemes: Set of all graphemes.
    :param loaded_phonemes: Set of all phonemes.
    :param all_langs: Set of all languages.
    :param exp: Experiment name.
    :return: Dictionary of settings to save to experiment directory. Used for training with DeepPhonemizer.
    """
    config_copy = master_config.copy()

    # remove extra validation information
    del config_copy['validation']

    # add graphemes, phonemes, and languages
    preprocessing = config_copy['preprocessing']
    preprocessing.update(
        {'text_symbols': list(loaded_graphemes), 'phoneme_symbols': list(loaded_phonemes), 'languages': all_langs})

    # update logging and data directories
    log_dir = config_copy['paths']['checkpoint_dir']
    data_dir = config_copy['paths']['data_dir']
    config_copy['paths']['checkpoint_dir'] = os.path.join("experiments", exp, log_dir)
    config_copy['paths']['data_dir'] = os.path.join("experiments", exp, data_dir)

    index = 0
    for text_symbol in config_copy['preprocessing']['text_symbols']:
        config_copy['preprocessing']['text_symbols'][index] = SingleQuotedScalarString(text_symbol)
        index = index + 1

    index = 0
    for phoneme_symbol in config_copy['preprocessing']['phoneme_symbols']:
        config_copy['preprocessing']['phoneme_symbols'][index] = SingleQuotedScalarString(phoneme_symbol)
        index = index + 1

    return config_copy


def get_all_langs(language_data: list[list[list[tuple[str, str, list[str]]] | str]]) -> list[str]:
    """
    Returns all unique languages from a list of languages and all its entries.

    :param language_data: Language list containing all languages and its entries as created from ``process_languages``.
    :return: Set of all unique languages.
    """
    all_langs = []
    for language in language_data:
        all_langs.append(language[1])

    return all_langs


def create_dp_sets(full_set: list[list[list[tuple[str, str, list[str]]] | str]], master_config: dict) -> tuple[list[tuple[str, str, list[str]] | str], list[tuple[str, str, list[str]]]]:
    """
    Creates DeepPhonemizer training and validation sets from full language data as created from ``process_languages``.

    :param full_set: Full language list containing all languages and their entries.
    :param master_config: Master configuration dictionary.
    :return: A tuple containing training and validation lists for use in DeepPhonemizer.
    """
    val_settings = master_config['validation']
    val_min, val_max, val_percentage = val_settings['validation_minimum'], val_settings['validation_maximum'], \
    val_settings['validation_percentage']

    print(f"val_minimum: {val_min}, val_max: {val_max}, val_percentage: {val_percentage}")

    training = []
    validation = []

    with tqdm(total=len(full_set)) as bar:
        for language in full_set:
            bar.set_description(f"Creating training and validation for {language[1]}")

            val_amounts = get_validation_amount(len(language[0]), val_min, val_max, val_percentage)

            for _ in range(val_amounts):
                selected_word = language[0].pop(language[0].index(random.choice(language[0])))
                validation.append(selected_word)

            for entry in language[0]:
                training.append(entry)

            bar.update(1)

    return training, validation


def get_validation_amount(num_entries: int, val_min: int, val_max: int, val_percentage: float) -> int:
    """
    Generates amount of entries that should be picked from each language in the global validation set.

    :param num_entries: Number of total entries in a language.
    :param val_min: Minimum number of entries that can be selected.
    :param val_max: Maximum number of entries that can be selected.
    :param val_percentage: Percentage of entries that will be selected, modified by ``val_min`` and ``val_max`` if out of range.
    :return:
    """
    validation_amounts = round(num_entries * val_percentage)

    if validation_amounts < val_min:
        return val_min
    elif validation_amounts > val_max:
        return val_max
    else:
        return validation_amounts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Helper script for DeepPhonemizer preprocessing.')
    parser.add_argument("config", type=str, help="Config file for DP training")
    parser.add_argument("linguistic", type=str, help="Linguistic file for DP preprocessing")
    parser.add_argument("exp_name", type=str, help="Experiment name")
    args = parser.parse_args()

    config = args.config
    linguistic = args.linguistic
    exp_name = args.exp_name
    config_file = load_config(config)
    out_dir = os.path.join("experiments", exp_name)
    os.mkdir(out_dir)

    print(
        f"\n- config: {os.path.abspath(config)}\n- linguistic: {os.path.abspath(linguistic)}\n- experiment: {exp_name}\n")
    print(f"Parsing linguistic file: {os.path.abspath(linguistic)}")
    languages = process_languages(load_config(linguistic))
    all_lang_names = get_all_langs(languages)

    print(f"\nProcessing graphemes and phonemes...")
    graphemes, phonemes = collate_symbols(languages)
    print(f"Total graphemes: {len(graphemes)}, Total phonemes: {len(phonemes)}\n")

    print(f"Creating experiment...")
    modified_config = create_dp_config(config_file, graphemes, phonemes, all_lang_names, exp_name)

    yaml = YAML(typ='rt')
    yaml.default_flow_style = None
    yaml.preserve_quotes = True

    with open(os.path.join(out_dir, "config.yaml"), 'w', encoding='utf-8') as f:
        yaml.dump(modified_config, f)

    print(modified_config)

    print("Selecting validation data...")
    training_set, validation_set = create_dp_sets(languages, config_file)

    print(f"\nValidating with {len(validation_set)} words\n")
    dp.preprocess.preprocess(config_file=os.path.join(out_dir, "config.yaml"),
                             train_data=training_set,
                             val_data=validation_set,
                             deduplicate_train_data=False)