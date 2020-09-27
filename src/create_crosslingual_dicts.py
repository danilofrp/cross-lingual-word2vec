import json
import time
import numpy as np
import pandas as pd

from tqdm import tqdm, trange
from pathlib import Path
from multiprocessing import Process, Queue
from googletrans import Translator

import utils

base_path = Path(".") / "data" 
paracrawl_pt_data_path = base_path / "parallel" / "paracrawl.en-pt" / "paracrawl.en-pt.pt"
paracrawl_en_data_path = base_path / "parallel" / "paracrawl.en-pt" / "paracrawl.en-pt.en"
en_pt_dict_path = base_path / "dicts" / "en-pt.json" 
pt_en_dict_path = base_path / "dicts" / "pt-en.json" 
translator = Translator()


def translate_words(queue, index, words, src_lang, dest_lang):
    try:
        translations = translator.translate(words, src = src_lang, dest = dest_lang)
        queue.put((index, [t.text for t in translations])) 
    except Exception as ex:
        print(ex)
        queue.put((index, words)) 


def translate_chunk(words, src_lang, dest_lang, n_processes):
    results = []
    q = Queue()
    processes = [
        Process(target=translate_words, args=(q, index, list(chunk), src_lang, dest_lang)) 
        for index, chunk in enumerate(np.array_split(words, n_processes))
    ]
    [p.start() for p in processes]
    [results.append(q.get()) for _ in processes]
    [p.join() for p in processes]
    translations = np.array([r[1] for r in sorted(results)]).flatten()
    restult_dict = dict(zip(words, translations))

    return restult_dict

def main():
    start_time = time.time()

    ordered_words = dict()
    print("Reading pt corpus")
    ordered_words["pt"] = utils.get_file_unique_word_counts(paracrawl_pt_data_path, nlines = None)
    ordered_words["pt"] = sorted(list(ordered_words["pt"].items()), key = lambda x: x[1], reverse=True)
    ordered_words["pt"] = [t[0] for t in ordered_words["pt"]]

    print("Reading en corpus")
    ordered_words["en"] = utils.get_file_unique_word_counts(paracrawl_en_data_path, nlines = None)
    ordered_words["en"] = sorted(list(ordered_words["en"].items()), key = lambda x: x[1], reverse=True)
    ordered_words["en"] = [t[0] for t in ordered_words["en"]]

    max_index = max(len(ordered_words["pt"]), len(ordered_words["en"]))
    # max_index = len(ordered_words["pt"])

    chunk_size = 2000
    n_processes = 100
    en_pt_dict = dict()
    pt_en_dict = dict()
    print("creating dicts")
    for i in trange(int(max_index/chunk_size) + 1):

        pt_words = ordered_words["pt"][chunk_size*i:chunk_size*(i+1)]
        if pt_words:
            temp_dict = translate_chunk(pt_words, "pt", "en", n_processes)
            pt_en_dict.update(temp_dict)
            
            with open(pt_en_dict_path, "wb") as f:
                f.write(json.dumps(pt_en_dict, indent=4, ensure_ascii=False).encode("utf8"))
        
        en_words = ordered_words["en"][chunk_size*i:chunk_size*(i+1)]
        if en_words:
            temp_dict = translate_chunk(en_words, "en", "pt", n_processes)
            en_pt_dict.update(temp_dict)

            with open(en_pt_dict_path, "wb") as f:
                f.write(json.dumps(en_pt_dict, indent=4, ensure_ascii=False).encode("utf8"))
        
    print(f"[ ] completed in {time.time() - start_time} seconds")
    
    

if __name__ == "__main__":
    print("start")
    main()