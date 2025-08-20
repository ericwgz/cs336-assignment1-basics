from pretokenization_example import pre_tokenize
import datasets
import heapq
from node_class import DoubleLinkedListWord 

if __name__ == '__main__':
    # Read the Tiny Stories dataset.
    # For real run, change the split to "all".
    corpus_path = "tiny_stories_train.txt"
    '''
    ds = datasets.load_dataset("roneneldan/TinyStories", split="validation")
    print("Start to load the dataset.")
    with open(corpus_path, "wb") as f:
        for story in ds:
            # For each story, convert it to the UTF-8 encoding.
            f.write(story["text"].encode("utf-8"))
            f.write("<|endoftext|>".encode("utf-8"))
    print("Dataset loaded and saved to tiny_stories_train.txt.")
    '''
    
    # Calling pretokenization to split the text into chunks and distribute into processes.
    word_count = pre_tokenize(corpus_path, num_processes=8)

    for i in range(10):
        print(f"Word: {list(word_count.keys())[i]}, Count: {word_count[list(word_count.keys())[i]]}")

    # Iterate the words to set up the priority queue that holds byte pair counts, and the words represented by double linked list, where each node is a byte, and it contains the 

    # Count each character
    token_count = {}
    # Count each pair of characters
    pair_count = {}
    token_to_word_list = {}
    for word, count in word_count.items():
        double_linked_list_word = DoubleLinkedListWord(word)
        for i in range(len(word)):
            character = word[i]
            token_count[character] = token_count.get(character, 0) + count
            if character not in token_to_word_list:
                token_to_word_list[character] = [] 
            token_to_word_list[character].append(double_linked_list_word)
            if (i != len(word) - 1):
                if (word[i], word[i + 1]) not in pair_count:
                    pair_count[(word[i], word[i + 1])] = 0
                pair_count[(word[i], word[i + 1])] = pair_count[(word[i], word[i+1])] + count

    for i in range(10):
        print(f"Word: {list(token_count.keys())[i]}, Count: {token_count[list(token_count.keys())[i]]}")
    for i in range(10):
        print(f"Word: {list(pair_count.keys())[i]}, Count: {pair_count[list(pair_count.keys())[i]]}")

    heap = []
    for pair, count in pair_count.items():
        heapq.heappush(heap, (-count, pair))
    
    count, pair = heapq.heappop(heap)
    print("First merge pair: (" + pair[0] + ", " + pair[1] + "), count: " + str(count))
    count, pair = heapq.heappop(heap)
    print("Second merge pair: (" + pair[0] + ", " + pair[1] + "), count: " + str(count))
    count, pair = heapq.heappop(heap)
    print("Third merge pair: (" + pair[0] + ", " + pair[1] + "), count: " + str(count))


        

    # 3. BPE training.

    # 4. Map-Reduce to count the number of tokens in each chunk.

    # 5. Adding the most common tokens to the vocabulary, adding the merge rules.

    # 6. Go back to step 4 until the vocabulary size is reached.