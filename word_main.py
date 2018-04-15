"""
    Inspired by https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
"""

from __future__ import print_function
import helper
import numpy as np
import random
import sys
from keras.models import load_model

"""
    Define global variables.
"""
WORD_SEQUENCE_LENGTH = 20
WORD_SEQUENCE_STEP = 1
PATH_TO_CORPUS = "leevi_corpus.txt"
EPOCHS = 5
DIVERSITY = 1.0

"""
    Read the corpus and get unique characters from the corpus.
"""
text = helper.read_corpus(PATH_TO_CORPUS)
words = text.split()
unique_words = helper.extract_characters(words)

"""
    Create sequences that will be used as the input to the network.
    Create next_chars array that will serve as the labels during the training.
"""
word_sequences, next_words = helper.create_word_sequences(words, WORD_SEQUENCE_LENGTH, WORD_SEQUENCE_STEP)
word_to_index, indices_word = helper.get_chars_index_dicts(unique_words)

# """
#     The network is not able to work with characters and strings, we need to vectorise.
# """
X, y = helper.vectorize(word_sequences, WORD_SEQUENCE_LENGTH, unique_words, word_to_index, next_words)

# """
#     Define the structure of the model.
# """
model = helper.build_model(WORD_SEQUENCE_LENGTH, unique_words)

# """
#     Train the model
# """

model.fit(X, y, batch_size=128, nb_epoch=EPOCHS)
# model = load_model("final.h5")  # you can skip training by loading the trained weights

for diversity in [0.2, 0.5, 1.0, 1.2]:
    print()
    print('----- diversity:', diversity)

    generated = []
    sentence = ['amalia', 'kamalia', 'tansseja']
    generated += sentence

    print('----- Generating with seed: "' + ' '.join(sentence))
    sys.stdout.write(' '.join(generated) + ' ')

    for i in range(WORD_SEQUENCE_LENGTH - len(sentence)):
        x = np.zeros((1, WORD_SEQUENCE_LENGTH, len(unique_words)))
        for t, word in enumerate(sentence):
            x[0, t, word_to_index[word]] = 1.

        predictions = model.predict(x, verbose=0)[0]
        next_index = helper.sample(predictions, diversity)

        next_word = indices_word[next_index]
        generated += next_word
        sentence.append(next_word)

        sys.stdout.write(next_word + ' ')
        sys.stdout.flush()
    print()



