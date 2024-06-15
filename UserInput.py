from flask import Blueprint, current_app
import re
import editdistance


userinput_bp = Blueprint('userinput_bp', __name__)


def find_correct_words_by_edit_distance(words_separated_by_space):
    # Lista per memorizzare le parole corrette
    correct_words = []

    for word in words_separated_by_space:
        found = False
        min_distance = float('inf')
        closest_word = None
        for token in current_app.tokenizer.get_vocab():

            distance = editdistance.eval(word, token)
            if distance == 0:
                correct_words.append(word)
                found = True
                break
            if distance < min_distance:
                min_distance = distance
                closest_word = token
        if closest_word and found == False :
            correct_words.append(closest_word)
    return correct_words

def identify_relevant_books(tokens, vectorizer, tfidf_matrix):
    relevant_books = []
    for token in tokens:
        if token.lower() in vectorizer.vocabulary_:
            token_key = token.lower()
        elif token.lower() + 'ã' in vectorizer.vocabulary_:
            token_key = token.lower() + 'ã'
        else:
            continue

        for i in range(len(current_app.corpora_book_names)):
            tfidf_value = tfidf_matrix.toarray()[i][vectorizer.vocabulary_[token_key]]
            if(tfidf_value > 0.018):
                relevant_books.append(current_app.corpora_book_names[i])
    return list(set(relevant_books))

def remove_g_prefix(strings):
    modified_strings = []
    for string in strings:
        if string == 'Ġ' or string == 'Ċ' or string=='ł' or string=='Ä':
            continue
        if string.startswith('Ġ') and len(string) > 1:
            modified_string = string[1:]
            modified_strings.append(modified_string)
        else:
            modified_strings.append(string)
    return modified_strings

def remove_punctuation(sentence):
    # Definisci un'espressione regolare per la punteggiatura di fine frase
    punctuation_pattern = re.compile(r'[.!?,;:]')
    # Rimuovi la punteggiatura di fine frase dalla frase
    sentence_without_punctuation = punctuation_pattern.sub('', sentence)
    return sentence_without_punctuation