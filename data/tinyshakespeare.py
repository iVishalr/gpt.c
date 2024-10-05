"""
Downloads and tokenizes the TinyShakespeare dataset.
- The file is downloaded from Github
- The tokenization is GPT-2 tokenizer with tiktoken

The output is written to a newly created data/ folder.
The script prints:

Saved 32768 tokens to data/tinyshakespeare_val.bin
Saved 305260 tokens to data/tinyshakespare_train.bin
"""

import os
import requests

import tiktoken
import numpy as np

from common import download_file, write_datafile

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tinyshakespeare")
enc = tiktoken.get_encoding("gpt2")
encoder = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})


def download():
    """Downloads the TinyShakespeare dataset to DATA_CACHE_DIR"""

    download_path = DATA_CACHE_DIR
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_filename = os.path.join(download_path, 'tinyshakespeare.txt')

    if not os.path.exists(data_filename):
        print(f'Downloading {data_url} to {data_filename}')
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

def tokenize():
    eot = enc._special_tokens['<|endoftext|>'] # end of text token
    data_filename = os.path.join(DATA_CACHE_DIR, 'tinyshakespeare.txt')
    
    text = None
    with open(data_filename, 'r') as file:
        text = file.read()

    if text is None:
        raise ValueError(f"{data_filename} might be empty.")

    # let's treat every person's statement in the dialog as a separate document
    text = "<|endoftext|>" + text
    text = text.replace('\n\n', '\n\n<|endoftext|>')
    
    # encode the text
    tokens = encoder(text)
    tokens_np = np.array(tokens, dtype=np.int32)
    # let's take the first 32,768 tokens as the validation split (~10%)
    val_tokens_np = tokens_np[:32768]
    train_tokens_np = tokens_np[32768:]
    # save to file
    val_filename = os.path.join(DATA_CACHE_DIR, "tinyshakespeare_val.bin")
    train_filename = os.path.join(DATA_CACHE_DIR, "tinyshakespeare_train.bin")
    
    write_datafile(val_filename, val_tokens_np)
    write_datafile(train_filename, train_tokens_np)

if __name__ == "__main__":
    download()
    tokenize()