import os
import requests
import numpy as np
from tqdm import tqdm

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

HEADERS_INFO = {
    "gpt-2": {
        "magic": 20240520,
        "version": 1,
        "token_dtype": np.uint32,
    },
    "llama-3": {
        "magic": 20240801,
        "version": 7,
        "token_dtype": np.uint32,
    },
}

def write_datafile(filename, toks, model_desc="gpt-2"):
    """
    Saves token data as a .bin file, for reading in C.
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    assert model_desc in ["gpt-2", "llama-3"], f"unknown model descriptor {model_desc}"
    info = HEADERS_INFO[model_desc]
    # construct the data (numpy array of tokens)
    toks_np = np.array(toks, dtype=np.int32)
    # write to file
    num_bytes = (len(toks) * toks_np.itemsize)
    print(f"writing {len(toks):,} tokens to {filename} ({num_bytes:,} bytes) in the {model_desc} format")
    with open(filename, "wb") as f:
        f.write(toks_np.tobytes())