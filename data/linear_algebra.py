
import os
import requests

import tiktoken
import numpy as np

from common import download_file, write_datafile

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "linalg")
enc = tiktoken.get_encoding("gpt2")
encoder = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})

def download():
    """Downloads the Linear Algebra Tex file from Github"""

    download_path = DATA_CACHE_DIR
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    data_url = "https://github.com/winitzki/linear-algebra-book/raw/refs/heads/master/linalg-src/linalg.tex"
    data_filename = os.path.join(download_path, 'linalg.txt')

    if not os.path.exists(data_filename):
        print(f'Downloading {data_url} to {data_filename}')
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")
    
def tokenize():
    eot = enc._special_tokens['<|endoftext|>'] # end of text token
    data_filename = os.path.join(DATA_CACHE_DIR, 'linalg.txt')

    text = None
    with open(data_filename, 'r', encoding="latin1") as file:
        text = file.read()
    
    if text is None:
        raise ValueError(f"{data_filename} might be empty.")

    # let's treat every paragraph in the tex file as separate document
    text = "<|endoftext|>" + text
    text = text.replace('\n\n', '\n\n<|endoftext|>')

    # encode the text
    tokens = encoder(text)
    tokens_np = np.array(tokens, dtype=np.int32)

    # let's take the first 34068 tokens as the validation split (~10%)
    val_tokens_np = tokens_np[:34068]
    train_tokens_np = tokens_np[34068:]

    # save to file
    val_filename = os.path.join(DATA_CACHE_DIR, 'linalg_val.bin')
    train_filename = os.path.join(DATA_CACHE_DIR, 'linalg_train.bin')

    write_datafile(val_filename, val_tokens_np)
    write_datafile(train_filename, train_tokens_np)


if __name__ == "__main__":
    download()
    tokenize()