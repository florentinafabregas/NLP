import random
import sys

#Dictionaries for char replacement
approx_dict_lower =  {
 
 'q': ['w', 'a', 's'],
 'w': ['q', 'e', 'a', 's', 'd'],
 'e': ['w', 'r', 's', 'd', 'f'],
 'r': ['e', 't', 'd', 'f', 'g'],
 't': ['r', 'y', 'f', 'g', 'h'],
 'y': ['t', 'u', 'g', 'h', 'j'],
 'u': ['y', 'i', 'h', 'j', 'k'],    
 'i': ['u', 'o', 'j', 'k', 'l'],
 'o': ['i', 'p', 'k', 'l'],
 'p': ['o', 'l'],
 'a': ['q', 'w', 's', 'x', 'z'],
 's': ['e', 'w', 'a', 'd', 'z', 'x'],
 'd': ['r', 'e', 's', 'f', 'x', 'c'],
 'f': ['t', 'r', 'd', 'g', 'c', 'v'],
 'g': ['y', 't', 'f', 'h', 'v', 'b'],
 'h': ['u', 'y', 'g', 'j', 'b', 'n'],
 'j': ['i', 'u', 'h', 'k', 'n', 'm'],
 'k': ['o', 'i', 'j', 'l', 'm', ','],
 'l': ['i', 'p', 'o', 'k', ',', '.'],
 'z': ['x', 'a', 's'],
 'x': ['s', 'd', 'z', 'c'],
 'c': ['d', 'f', 'x', 'v'],
 'v': ['f', 'g', 'c', 'b'],
 'b': ['g', 'h', 'v', 'n'],
 'n': ['h', 'j', 'b', 'm'],
 'm': ['j', 'k', 'n', ','],
 ',': ['k', 'l', 'm', '.'],
 '.': ['l', ',']
    }

approx_dict_upper = {
    'Q': ['W', 'A', 'S'],
 'W': ['Q', 'E', 'A', 'S', 'D'],
 'E': ['W', 'R', 'S', 'D', 'F'],
 'R': ['E', 'T', 'D', 'F', 'G'],
 'T': ['R', 'Y', 'F', 'G', 'H'],
 'Y': ['T', 'U', 'G', 'H', 'J'],
 'U': ['Y', 'I', 'H', 'J', 'K'],
 'I': ['U', 'O', 'J', 'K', 'L'],
 'O': ['I', 'P', 'K', 'L'],
 'P': ['O', 'L'],
 'A': ['Q', 'W', 'S', 'X', 'Z'],
 'S': ['E', 'W', 'A', 'D', 'Z', 'X'],
 'D': ['R', 'E', 'S', 'F', 'X', 'C'],
 'F': ['T', 'R', 'D', 'G', 'C', 'V'],
 'G': ['Y', 'T', 'F', 'H', 'V', 'B'],
 'H': ['U', 'Y', 'G', 'J', 'B', 'N'],
 'J': ['I', 'U', 'H', 'K', 'N', 'M'],
 'K': ['O', 'I', 'J', 'L', 'M', ','],
 'L': ['I', 'P', 'O', 'K', ',', '.'],
 'Z': ['X', 'A', 'S'],
 'X': ['S', 'D', 'Z', 'C'],
 'C': ['D', 'F', 'X', 'V'],
 'V': ['F', 'G', 'C', 'B'],
 'B': ['G', 'H', 'V', 'N'],
 'N': ['H', 'J', 'B', 'M'],
 'M': ['J', 'K', 'N', ','],
 ',': ['K', 'L', 'M', '.'],
 '.': ['L', ',']}

# The two functions necesary for capitalization swap
def capitalization_swap(sentence, rate):
    #Collect index of each starting letter of the sentence
    #Assume sentence starts with a word (which it does in english)
    word_start_idx = [0]
    for idx, letter in enumerate(sentence):
        if letter == " ":
            word_start_idx.append(idx+1)
    #Select which words, based on the 'rate' threshold, that are to be altered
    selected_word_index = []
    for item in word_start_idx:
        if random.random() <= rate:
            selected_word_index.append(item)
    #At the the specific indexes, swap the capitalization of said letter
    for idx in selected_word_index:
        swapped_cap = sentence[idx].swapcase()
        sentence = sentence[:idx] + swapped_cap + sentence[idx+1:]
    return sentence

#This is the actual function to use on the full text (have the text be a list of strings)
def swap_capitalization_corpus(texts, rate):
    # Applies the capitalization swap to a list of sentences and returns a new list where the sentences have been altered
    altered_corpus = []
    for sentence in texts:
        #Easy way to skip empty lines =)
        if len(sentence) != 0:
            altered_corpus.append(capitalization_swap(sentence,rate))
    return altered_corpus

# The two functions necessary for character swap
def character_swap(sentence,rate):
    #Swaps a character with the following character
    sentence_indexes = list(range(len(sentence)))
    chosen_indexes = random.sample(sentence_indexes,int(rate*len(sentence_indexes)))
    for idx in chosen_indexes:
        #For all except last two indexes, swap two characters at index chosen
        if idx <= len(sentence)-3:
            sentence = sentence[:idx] + sentence[idx+1] + sentence[idx] + sentence[idx+2:]
        #For second last index, no need to do "rest" as we are at the end
        elif idx == len(sentence)-2:
            sentence = sentence[:idx] + sentence[idx+1] + sentence[idx]
    return sentence

#This is the actual function to use on the full text (have the text be a list of strings)
def swap_character_corpus(texts,rate):
    # Applies the character swap to a list of sentences, returning the altered list
    altered_corpus = []
    for sentence in texts:
        #Easy way to skip empty lines =)
        if len(sentence) != 0:
            altered_corpus.append(character_swap(sentence,rate))
    return altered_corpus  

# The two functions necessary for character removal
def character_removal(sentence,rate):
    #Swaps a character with the following character
    sentence_indexes = list(range(len(sentence)))
    chosen_indexes = random.sample(sentence_indexes,int(rate*len(sentence_indexes)))
    #Initialize a counter. Every time the sentence is shortened, this counter is incremented. The counter is then subtracted
    # from the index, so that the index represents the correct spot for the altered string.
    cull_counter = 0
    chosen_indexes.sort()
    for idx in chosen_indexes:
        idx = idx - cull_counter
        #Edge case 1: if index is 0, just take entire list minus first entry
        if idx == 0:
            sentence = sentence[1:]
            cull_counter += 1            
        #For all except last index, concatinate sentence before idx of removel with after index of removal
        elif idx <= len(sentence)-2:
            sentence = sentence[:idx] + sentence[idx+1:]
            cull_counter += 1
        #For second last index, take the entire list minus last entry
        elif idx == len(sentence)-1:
            sentence = sentence[:-1]
    return sentence

#This is the actual function to use on the full text (have the text be a list of strings)
def remove_character_corpus(texts,rate):
    # Applies the character removal to a list of sentences, returning the altered list
    altered_corpus = []
    for sentence in texts:
        # Skip empty lines =) 
        if len(sentence) != 0:
            altered_corpus.append(character_removal(sentence,rate))
    return altered_corpus

#Code for the character replacement, requires that the dictionary above is loaded in ofc
def character_replacement(sentence,rate):
    #swaps a char with a neighbor, based on qwerty keyboard
    sentence_indexes = list(range(len(sentence)))
    chosen_indexes = random.sample(sentence_indexes,int(rate*len(sentence_indexes)))
    for idx in chosen_indexes:
        current_letter = sentence[idx]
        if current_letter in approx_dict_lower:
            #selects one of the neighbors
            chosen_letter = random.sample(approx_dict_lower[current_letter],1)
            actual_letter = chosen_letter[0]
            sentence = sentence[:idx] + actual_letter + sentence[idx+1:]
        elif current_letter in approx_dict_upper:
            chosen_letter = random.sample(approx_dict_upper[current_letter],1)
            actual_letter = chosen_letter[0]
            sentence = sentence[:idx] + actual_letter + sentence[idx+1:]
        else:
            continue
    return sentence

#Self explanatory at this point
def character_replacement_corpus(texts,rate):
    #applies the char replacement to list of sentences, returning altered list
    altered_corpus = []
    for sentence in texts:
        # Skip empty lines =) 
        if len(sentence) != 0:
            altered_corpus.append(character_replacement(sentence,rate))
    return altered_corpus

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Please provide input.txt file noise type and rate')
        
    else:
        
        input_path = sys.argv[1]
        noise_type = sys.argv[2]
        rate = float(sys.argv[3])

        with open(input_path, 'r') as f:
            lines = f.readlines()

        sentences = []
        sentence = []
        for line in lines:
            line = line.strip()
            if line:  # If the line is not empty
                sentence.append(line.split('\t'))  # Split into word and tag
            else:  # If the line is empty, indicating end of a sentence
                if sentence:
                    sentences.append(sentence)
                sentence = []

        if sentence:
            sentences.append(sentence)

        altered_sentences = []
        for sentence in sentences:
            altered_sentence = []
            for word, ner_tag in sentence: 
                if noise_type == 'capitalization_swap':
                    altered_word = capitalization_swap(word, rate)
                elif noise_type == 'character_swap':
                    altered_word = character_swap(word, rate)
                elif noise_type == 'character_removal':
                    altered_word = character_removal(word, rate)
                elif noise_type == 'character_replacement':
                    altered_word = character_replacement(word, rate)
            
                altered_sentence.append((altered_word, ner_tag))
            altered_sentences.append(altered_sentence)

        # Write the altered data to a new .txt file
        output_file = f'out/noisy_file/{noise_type}_rate_{rate}.txt'
        with open(output_file, 'w') as f:
            for sentence in altered_sentences:
                for word, ner_tag in sentence:
                    f.write(f'{word}\t{ner_tag}\n')
                f.write('\n') 