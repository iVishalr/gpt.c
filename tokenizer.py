import argparse
import os
import sys
import struct

import torch
import tiktoken

from typing import List


def write_tokenizer(encoder: tiktoken.Encoding, filename: str):
    if os.path.exists(filename):
        raise FileExistsError(f"{filename} already exists!")
    
    dirname = os.path.dirname(filename)
    _filename, ext = os.path.splitext(os.path.basename(filename))

    if ext != '.bin':
        raise ValueError(f"Expected filename to have .bin extension. Got {ext}")
    
    if dirname and not os.path.exists(filename):
        os.makedirs(dirname)

    n = encoder.max_token_value + 1
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240415
    header[1] = n
    header[2] = encoder.eot_token # EOT Token
    
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes())
        for i in range(n):
            b = encoder.decode_bytes([i])
            length = len(b)
            assert length < 256, f"Token length exceeds 255: {length}"
            file.write(struct.pack("<B", length)) # Write the length as a 1-byte unsigned integer
            file.write(b) # Write the actual bytes
    
    print(f"Saved tokenizer at {filename}")


def encode(encoder: tiktoken.Encoding, prompt: str) -> list:
    prompt = "<|endoftext|>" + prompt
    tokens = encoder.encode(text=prompt, allowed_special="all")
    return tokens


def decode(encoder: tiktoken.Encoding, tokens: List[int]) -> str:
    return encoder.decode(tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="tokenizer.py",
    )
    parser.add_argument("-t","--tokenizer", type=str, required=False, default="gpt2", help="Pass name of tokenizer.")
    parser.add_argument("-p","--prompt", type=str, default="", required=False, help="Pass the prompt for encoding.")
    parser.add_argument("-e","--encode", required=False, action="store_true", help="Pass this flag for encoding the prompt.")
    parser.add_argument("-d","--decode", required=False, action="store_true", help="Pass this flag for decoding the prompt.")
    parser.add_argument("-s","--save", required=False, action="store_true", help="Pass this flag for saving the tokenizer.")
    parser.add_argument("-f","--file", type=str, default="tokenizer.bin", required=False, help="Path to the file (.bin extension) to save the tokenizer.")
    args = parser.parse_args()

    if args.encode and args.decode:
        raise argparse.ArgumentError(None, "Pass either --encode or --decode flags, not both.")
    
    if len(args.prompt) == 0 and not args.save:
        raise argparse.ArgumentError(None, f"Expected prompt to be of atleast length 1. Got {len(args.prompt)}")
    
    if not args.encode and not args.decode and not args.save:
        raise argparse.ArgumentError(None, f"Pass either --encode, --decode or --save flags.")
    
    encoder = tiktoken.get_encoding(args.tokenizer)

    if args.save:
        write_tokenizer(encoder, args.file)
        exit(0)
    
    if args.encode:
        print(encode(encoder, args.prompt))

    if args.decode:
        prompt = args.prompt
        prompt = prompt.strip()
        if '[' == prompt[0] and ']' in prompt:
            prompt = prompt[1:-1]
        if ', ' in prompt:
            prompt = prompt.replace(', ', ',')
        
        prompt = prompt.split(",")
        prompt = [int(token) for token in prompt]
        print(decode(encoder, prompt))

