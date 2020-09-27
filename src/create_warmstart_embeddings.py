import sys
import csv
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path

import utils

embed_dim = 50

data_path = Path("data")
pretrained_embeddings_path = data_path/"pretrained-embeddings"/"glove.6B"/f"glove.6B.{embed_dim}d.txt"

paracrawl_en_data_path = data_path / "parallel" / "paracrawl.en-pt" / "paracrawl.en-pt.en"
paracrawl_pt_data_path = data_path / "parallel" / "paracrawl.en-pt" / "paracrawl.en-pt.pt"

en_pt_dict_path = data_path / "dicts" / "en-pt.json"
pt_en_dict_path = data_path / "dicts" / "pt-en.json"


def glove2dict(glove_filename):
    with open(glove_filename, encoding="utf8") as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in tqdm(reader, total = 400_000)}
    return embed


def main():
    print("[ ] Loading EN unique words")
    en_word_counts = utils.get_file_unique_word_counts(paracrawl_en_data_path)
    total_en_ocourrences = sum(en_word_counts.values())
    print("[ ] Loading PT unique words")
    pt_word_counts = utils.get_file_unique_word_counts(paracrawl_pt_data_path)
    total_pt_ocourrences = sum(pt_word_counts.values())

    print(f"[ ] Loading pre-trained embeddings ({embed_dim}d)")
    pretrained_embeddings = glove2dict(pretrained_embeddings_path)
    print(f"[ ] Finished loading embeddings ({embed_dim}d)")

    warm_started_embeddings = dict()
    
    matches = 0
    covered_occourences = 0
    print("[ ] Building warm start for EN words")
    for word, count in tqdm(en_word_counts.items(), total = len(en_word_counts)):
        if word in pretrained_embeddings:
            warm_started_embeddings[word] = pretrained_embeddings[word]
            matches += 1
            covered_occourences += count
        else:
            warm_started_embeddings[word] = 2*np.random.rand(embed_dim) - 1
    print("[ ] Finished warm start for EN words")
    print(f"[ ] Matched {matches} EN words ({100*matches/len(en_word_counts):.2f}% of words, {100*covered_occourences/total_en_ocourrences:.2f}% of occourences)")


    matches = 0
    covered_occourences = 0
    print("[ ] Building warm start for PT words")
    for word, count in tqdm(pt_word_counts.items(), total = len(pt_word_counts)):
        translation = utils.translator.translate(word, "pt", "en")
        if translation in pretrained_embeddings:
            warm_started_embeddings[word] = pretrained_embeddings[translation]
            matches += 1
            covered_occourences += count
        else:
            warm_started_embeddings[word] = 2*np.random.rand(embed_dim) - 1
    print("[ ] Finished warm start for PT words")
    print(f"[ ] Matched {matches} PT words ({100*matches/len(pt_word_counts):.2f}% of words, {100*covered_occourences/total_pt_ocourrences:.2f}% of occourences)")


    print(f"[ ] Writing file")
    warm_started_embeddings_path = data_path/"warm-started-embeddings"/f"glove.6B.en-pt.{embed_dim}d.txt"
    with open(warm_started_embeddings_path, "w", encoding="utf8") as f:
        for key, value in tqdm(warm_started_embeddings.items(), total = len(warm_started_embeddings)):
            line = key + " " + str.join(" ", [str(v) for v in value]) + "\n"
            f.write(line)
    
    print(f"[ ] Done!")


if __name__ == "__main__":
    main()