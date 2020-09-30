import time
import matplotlib.pyplot as plt

from pathlib import Path
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec

import utils


embed_dim = 300
epochs = 50

n_lines = 2809381
data_path = Path("data")
warmstarted_embeddings_path = data_path/"warm-started-embeddings"/f"glove.6B.en-pt.{embed_dim}d.txt"

paracrawl_crosslingual_all_data_path = data_path / "parallel" / "paracrawl.en-pt" / "paracrawl.crosslingual.en-pt.all"

train_log_path = data_path / "train_logs" / f"loss_warm_started_{embed_dim}d_{epochs}epochs_{int(time.time())}.txt"
checkpoints_path = data_path / "checkpoints"
results_path = data_path / "results" / f"word2vec_warmstarted_trained_embeddings_{embed_dim}d_{epochs}epochs.txt"



class ModelCheckpoint(CallbackAny2Vec):
    def __init__(self, checkpoints_path):
        self.epoch = 0
        self.checkpoints_path = checkpoints_path

    def on_epoch_end(self, model):
        if (self.epoch + 1) % 2 == 0:
            print("[ ] Saving checkpoint...")
            filepath = self.checkpoints_path/f"word2vec-warmstart-{model.trainables.layer1_size}d-epoch{self.epoch + 1}.model"
            with open(filepath, "w") as f:
                filename = f.name
                model.save(filename)
            
        if (self.epoch + 1) % 2 == 0:
            filepath = data_path/"results"/f"word2vec_warmstarted_trained_embeddings_{embed_dim}d_epoch{self.epoch + 1}.txt"
            model.wv.save_word2vec_format(str(filepath))
        
        self.epoch += 1

class LossLogger(CallbackAny2Vec):
    def __init__(self, train_log_path):
        self.epoch = 0
        self.batch = 0
        self.open_batches = []
        self.train_log_path = train_log_path
        self.epoch_start_time = time.time()

    def on_train_begin(self, model):
        print("[ ] Starting Word2Vec training...")
        self.start_time = time.time()

    def on_train_end(self, model):
        print(f"[ ] Word2Vec finished training in {time.time() - self.start_time} seconds")
        self.start_time = time.time()

    def on_batch_begin(self, model):
        print(f"[ ] Starting epoch {self.epoch}, batch {self.batch}...")
        self.open_batches.append(self.batch)
        self.batch += 1

    def on_batch_end(self, model):
        completed_batch = "?"
        try:
            completed_batch = self.open_batches.pop(0)
        except:
            pass

        print(f'[ ] Batch {completed_batch} of epoch {self.epoch} finished')

    def on_epoch_begin(self, model):
        print(f"[ ] Starting epoch {self.epoch}...")
        self.batch = 0
        self.open_batches = []
        self.epoch_start_time = time.time()

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print(f'[ ] Epoch {self.epoch} finished. Time elapsed: {int(time.time() - self.epoch_start_time)}s. Loss: {loss:.3f}')

        with open(self.train_log_path, "a") as f:
            f.write(str(loss) + "\n")
        
        self.epoch += 1


def main():
    print("[ ] Loading warmstarted embeddings...")
    warmstart_embeddings = KeyedVectors.load_word2vec_format(warmstarted_embeddings_path, binary=False)
    
    print("[ ] Initializing model...")
    model = Word2Vec(size = embed_dim, 
                     iter = epochs,
                     min_count = 10, 
                     compute_loss = True, 
                     callbacks = [ModelCheckpoint(checkpoints_path), LossLogger(train_log_path)]
                    )

    print("[ ] Building vocab...")
    time.sleep(1)
    corpus = list(utils.single_file_corpus(paracrawl_crosslingual_all_data_path))
    model.build_vocab(corpus)
    total_examples = model.corpus_count

    print("[ ] Updating vocab...")
    model.build_vocab([list(warmstart_embeddings.vocab.keys())], update=True)

    print("[ ] Warm-starting embeddings...")
    model.intersect_word2vec_format(warmstarted_embeddings_path, binary=False, lockf=1.0)

    print("[ ] Training model...")
    model.train(corpus, total_examples=total_examples, epochs=model.epochs, compute_loss = True)

    print("[ ] Saving model results...")
    model.wv.save_word2vec_format(str(results_path))


if __name__ == "__main__":
    main()



    # paracrawl_crosslingual_all_data_path = data_path / "parallel" / "paracrawl.en-pt" / "paracrawl.crosslingual.en-pt.all"
    # paracrawl_crosslingual_all_data_sample_path = data_path / "parallel" / "paracrawl.en-pt" / "paracrawl.crosslingual.en-pt.all.sample"
    # lines_written = 0
    # with open(paracrawl_crosslingual_all_data_sample_path, "w", encoding = "utf8") as f:
    #     for line in utils.single_file_corpus(paracrawl_crosslingual_all_data_path):
    #         f.write(str.join(" ", line) + "\n")
    #         lines_written += 1

    #         if lines_written == 10_000:
    #             break