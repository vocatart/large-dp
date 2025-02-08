import argparse
import os
import torch
from dp.phonemizer import Phonemizer
from dp.preprocessing.text import Preprocessor, LanguageTokenizer, SequenceTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Helper script for tracing DeepPhonemizer models.')
    parser.add_argument("exp_name", type=str, help="Experiment name")
    args = parser.parse_args()

    # the newest ver of torch does not like some of the DP stuff
    torch.serialization.add_safe_globals([Preprocessor])
    torch.serialization.add_safe_globals([LanguageTokenizer])
    torch.serialization.add_safe_globals([SequenceTokenizer])

    out_dir = os.path.join("artifacts", args.exp_name + ".pt")
    ts_model = torch.jit.script(Phonemizer.from_checkpoint(os.path.join("experiments", args.exp_name, "logs", "best_model.pt")).predictor.model)

    torch.jit.save(ts_model, out_dir)

    print("Model saved to {}".format(out_dir))

