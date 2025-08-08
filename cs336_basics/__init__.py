import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

# 1. Read the Tiny Stories dataset.

import tensorflow_datasets as tfds

# The code below loads all dataset splits. You can also load a specific split
# by passing split="chosen_split_name".
# Our guide on using splits is here: https://www.tensorflow.org/datasets/splits.
ds = tfds.load("roneneldan__tinystories/default", data_dir=tfds.HF_MERLION_PLACER_DIR, split="all")
for ex in ds.take(5):
  print(ex)


# 2. For each story, convert it to the UTF-8 encoding.

# 3. Calling pretokenization to split the text into chunks and distribute into processes.

# 4. BPE training.

# 5. Map-Reduce to count the number of tokens in each chunk.

# 6. Adding the most common tokens to the vocabulary, adding the merge rules.

# 7. Go back to step 4 until the vocabulary size is reached. 