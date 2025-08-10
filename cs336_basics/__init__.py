from pretokenization_example import pre_tokenize
# 1. Read the Tiny Stories dataset.

# 2. For each story, convert it to the UTF-8 encoding.
 
# 3. Calling pretokenization to split the text into chunks and distribute into processes.
pre_tokenize("tiny_stories.txt", num_processes=8)

# 4. BPE training.

# 5. Map-Reduce to count the number of tokens in each chunk.

# 6. Adding the most common tokens to the vocabulary, adding the merge rules.

# 7. Go back to step 4 until the vocabulary size is reached.