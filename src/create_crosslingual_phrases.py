import time
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from multiprocessing import Process, Queue

import utils

def create_crosslingual_phrase(queue, phrases, src_lang, dest_lang, agressiveness, enforce_match):
    queue.put((phrases[0], utils.create_crosslingual_phrase(phrases[1], src_lang, dest_lang, agressiveness, enforce_match)))


def main():
    chunk_size = 100
    start_time = time.time()
    # for chunk in tqdm(utils.read_paracrawl_in_chunks(chunk_size = chunk_size), total = int(199/chunk_size) + 1):
    for chunk in tqdm(utils.read_paracrawl_in_chunks(chunk_size = chunk_size), total = int(utils.n_lines/chunk_size) + 1):
        # print(f"[+] processing chunk {list(chunk.keys())[0]/chunk_size + 1}")
        df = utils.filechunk_to_dataframe(chunk, split = False, pad = False)

        for src_lang, dest_lang in [("en", "pt"), ("pt", "en")]:

            results = df.apply(lambda row: utils.create_crosslingual_phrase(row, src_lang, dest_lang, agressiveness = 1, enforce_match = True), axis = 1)
            
            file_name = Path(".") / "data" / "parallel" / "paracrawl.en-pt" / f"paracrawl.crosslingual.{src_lang}-{dest_lang}"
            with open(file_name, "ab") as f:
                f.writelines([f"{r}\n".encode('utf8') for r in results.values])
    print(f"[ ] completed in {time.time() - start_time} seconds")
    
    

if __name__ == "__main__":
    try:
        (Path(".") / "data" / "parallel" / "paracrawl.en-pt" / f"paracrawl.crosslingual.en-pt").unlink()
    except:
        pass
    try:
        (Path(".") / "data" / "parallel" / "paracrawl.en-pt" / f"paracrawl.crosslingual.pt-en").unlink()
    except:
        pass

    main()