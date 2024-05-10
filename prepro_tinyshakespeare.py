"""
Downloads and tokenizes the TinyShakespeare dataset.
- The file is downloaded from Github
- The tokenization is GPT-2 tokienizer with tiktoken

The output is written to a newly created data/ folder.
The script prints:

Saved 32768 tokens to data/tiny_shakespeare_val.bin
Saved 305260 tokens to data/tiny_shakespare_train.bin
"""

import os
import requests

import tiktoken
import numpy as np

from tqdm import tqdm

DATA_CACHE_DIR = "data"
DATASET = 'tiny_shakespeare'
enc = tiktoken.get_encoding("gpt2")
encoder = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})

def download_file(url: str, filename: str, chunk_size = 1024):
    """Helper function to download a file from a given url"""

    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(filename, "wb") as file, tqdm(
        desc=filename,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def download():
    """Downloads the TinyShakespeare dataset to DATA_CACHE_DIR"""

    download_path = os.path.join(DATA_CACHE_DIR, DATASET)
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_filename = os.path.join(download_path, 'tiny_shakespeare.txt')

    if not os.path.exists(data_filename):
        print(f'Downloading {data_url} to {data_filename}')
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

def tokenize():
    eot = enc._special_tokens['<|endoftext|>'] # end of text token
    data_filename = os.path.join(DATA_CACHE_DIR, DATASET, 'tiny_shakespeare.txt')
    
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
    val_filename = os.path.join(DATA_CACHE_DIR, DATASET, "tiny_shakespeare_val.bin")
    train_filename = os.path.join(DATA_CACHE_DIR, DATASET, "tiny_shakespeare_train.bin")
    with open(val_filename, "wb") as f:
        f.write(val_tokens_np.tobytes())
    with open(train_filename, "wb") as f:
        f.write(train_tokens_np.tobytes())
    # prints
    print(f"Saved {len(val_tokens_np)} tokens to {val_filename}")
    print(f"Saved {len(train_tokens_np)} tokens to {train_filename}")

if __name__ == "__main__":
    download()
    tokenize()