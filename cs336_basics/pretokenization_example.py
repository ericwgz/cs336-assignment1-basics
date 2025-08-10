import os
from datasets import load_dataset
from typing import BinaryIO
import multiprocessing
import regex as re

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def regex_split(chunk: str):
    """
    Pre-tokenize a chunk of text.
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens = re.findall(PAT, chunk)
    print(f"Chunk size: {len(chunk)}")
    print(f"Number of pre-tokens: {len(pre_tokens)}")
    print("First 5 pre-tokens:")
    for i in range(min(len(pre_tokens), 5)):
        print(pre_tokens[i])
    return pre_tokens


def pre_tokenize(corpus_path ,num_processes: int = 8):   
    with open("tiny_stories.txt", "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

        with multiprocessing.Pool(processes=num_processes) as pool:
            list_of_pretokens = pool.map(regex_split, chunks)

        pre_token_sum = [pretoken for pretokens in list_of_pretokens for pretoken in pretokens]
        print(f"Total number of pre-tokens: {len(pre_token_sum)}")