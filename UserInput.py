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
        for token in current_app.vocabulary:

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

def identify_relevant_books(tokens):
    relevant_books = set()  # Usa un set per evitare duplicati
    # Pre-converti la matrice TF-IDF in array per l'accesso più veloce
    tfidf_matrix_array = current_app.tfidf_matrix.toarray()
    
    # Converti tutti i token a minuscolo
    lower_tokens = [token.lower() for token in tokens]
    
    for token in lower_tokens:
        if token in current_app.vectorizer.vocabulary_:
            token_index = current_app.vectorizer.vocabulary_[token]
            for i in range(len(current_app.corpora_book_names)):
                tfidf_value = tfidf_matrix_array[i][token_index]
                if tfidf_value > 0.015:
                    relevant_books.add(current_app.corpora_book_names[i])
    
    return list(relevant_books)

def remove_punctuation(sentence):
    # Definisci un'espressione regolare per la punteggiatura di fine frase
    punctuation_pattern = re.compile(r'[.!?,;:]')
    # Rimuovi la punteggiatura di fine frase dalla frase
    sentence_without_punctuation = punctuation_pattern.sub('', sentence)
    return sentence_without_punctuation