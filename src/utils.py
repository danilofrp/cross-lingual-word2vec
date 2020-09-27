import re
import csv
import time
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from functools import partial
from itertools import chain

from translator import Translator

data_path = Path("data") 
paracrawl_pt_data_path = data_path / "parallel" / "paracrawl.en-pt" / "paracrawl.en-pt.pt"
paracrawl_en_data_path = data_path / "parallel" / "paracrawl.en-pt" / "paracrawl.en-pt.en"
en_pt_dict_path = data_path / "dicts" / "en-pt.json"
pt_en_dict_path = data_path / "dicts" / "pt-en.json"
n_lines = 2809381


random_generator = np.random.default_rng()
remove_punct_regex = re.compile(r"[^\w\s]")
remove_punct = partial(remove_punct_regex.sub, repl = "")
translator = Translator(en_pt_dict_path, pt_en_dict_path)


def get_file_unique_words(file_path):
    with open(file_path, "rb") as f:
        unique_words = set(chain(*(remove_punct(string = line.decode("utf8").replace("\n", "").lower()).split() for line in tqdm(f, total = n_lines) if line)))
    return unique_words

def get_file_unique_word_counts(file_path, nlines = None):
    word_counts = dict()
    lines_read = 0
    with open(file_path, "rb") as f:
        for line in tqdm(f, total = n_lines):
            for word in remove_punct(string = line.decode("utf8").replace("\n", "").lower()).split():
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
            lines_read += 1
            if nlines and lines_read >= nlines:
                break
    return word_counts

def glove2dict(glove_filename):
    with open(glove_filename, encoding="utf8") as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in tqdm(reader, total = 400_000)}
    return embed

def read_paracrawl_in_chunks(en_data_path = paracrawl_en_data_path, pt_data_path = paracrawl_pt_data_path, chunk_size = 100, offset = 0):
    outer_break = False
    index = offset
    with open(en_data_path, "rb") as en_file, open(pt_data_path, "rb") as pt_file:
        while True:
            for _ in range(offset):
                en_file.readline()
                pt_file.readline()

            lines = {}
            for _ in range(chunk_size):
                en_data = en_file.readline()
                pt_data = pt_file.readline()
                if not en_data or not pt_data:
                    outer_break = True
                    break
                lines[index] = {
                    "en": remove_punct(string = en_data.decode("utf8").replace("\n", "").lower()),
                    "pt": remove_punct(string = pt_data.decode("utf8").replace("\n", "").lower())
                }
                index += 1
            if lines:
                yield lines
            if outer_break:
                break


def filechunk_to_dataframe(file_chunk, split = False, pad = False):
    df = pd.DataFrame(file_chunk).T
    if split:
        df.loc[:, "en"] = df.loc[:, "en"].str.split()
        df.loc[:, "pt"] = df.loc[:, "pt"].str.split()
    if pad:
        max_phrase_size = max(df["en"].apply(len).max(), df["pt"].apply(len).max())
        df["en"] = df["en"].apply(lambda v: v + ["<pad_token>"]*(max_phrase_size - len(v)))
        df["pt"] = df["pt"].apply(lambda v: v + ["<pad_token>"]*(max_phrase_size - len(v)))
    return df


def read_paracrawl(nrows = None):
    df_en = pd.read_csv(paracrawl_en_data_path, sep = "<placeholder>", header = None, names = ["en"], encoding="utf8", nrows = nrows)
    df_pt = pd.read_csv(paracrawl_pt_data_path, sep = "<placeholder>", header = None, names = ["pt"], encoding="utf8", nrows = nrows)
    df = pd.concat([df_en, df_pt], axis = 1)
    del df_en, df_pt
    df["en"] = df["en"].apply(lambda s: remove_punct(string = s))
    df["pt"] = df["pt"].apply(lambda s: remove_punct(string = s))
    return df


def create_crosslingual_phrase(phrases, src_lang, dest_lang, agressiveness = 0.5, enforce_match = True):
    # print(f"[ ] getting crosslingual rep for phrase {phrases[src_lang]}")
    original = phrases[src_lang].lower().split() 
    target = phrases[dest_lang]
    new_phrase = original.copy()
    indexes_to_translate = random_generator.choice(len(original), size = int(agressiveness*len(original)), replace=False)
    words_to_translate = [original[i] for i in indexes_to_translate]
    translations = translator.bulk_translate(words_to_translate, src_lang, dest_lang)
    
    for index, translation in zip(indexes_to_translate, translations):
        if not enforce_match or translation in target:
            new_phrase[index] = translation.lower()
    # print(f"[ ] done getting crosslingual rep for phrase {phrases[src_lang]}")

    return str.join(" ", new_phrase)
