from flask import Flask, render_template, request
from tokenizers import Tokenizer
from datasheet import datasheet_bp
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence import Sentence
from transformers import pipeline

import editdistance
import nltk
import re
import os

app = Flask(__name__)

app.register_blueprint(datasheet_bp)

app.corpora = []  # Lista per mantenere il testo di ciascun file
app.corpora_book_names = []
# Directory contenente i file di testo per l'addestramento del token learner
corpora_dir = './corpora'
app.tokenizer = Tokenizer.from_file("./data/newtokenizer.json")
qa_model = pipeline("question-answering")

# Lista delle stopwords nella lingua inglese
stop_words = set(stopwords.words('english'))

# Inizializza il lemmatizzatore
lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# Funzione per leggere i file di testo nel corpus e aggiornare il token learner
def train_token_learner(tokenizer, corpora_dir):
    
    # Itera attraverso i file di testo nel corpus
    for filename in os.listdir(corpora_dir):
        if filename.endswith('.raw') or filename.endswith('.txt'):
            file_path = os.path.join(corpora_dir, filename)
            # Leggi il contenuto del file di testo
            with open(file_path, 'r', encoding='utf-8') as file:
                corpus = file.read()
                app.corpora.append(corpus)  # Aggiungi il testo alla lista
                app.corpora_book_names.append(filename)
    # Aggiorna il token learner con il testo dei file
    tokenizer.train_from_iterator(app.corpora)

# Addestra il token learner con il corpus di testo
train_token_learner(app.tokenizer, corpora_dir)

app.tokenizer.save("./data/newtokenizer.json")
app.tokenizer = Tokenizer.from_file("./data/newtokenizer.json")

vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = vectorizer.fit_transform(app.corpora)

def remove_punctuation(sentence):
    # Definisci un'espressione regolare per la punteggiatura di fine frase
    punctuation_pattern = re.compile(r'[.!?,;:]')
    # Rimuovi la punteggiatura di fine frase dalla frase
    sentence_without_punctuation = punctuation_pattern.sub('', sentence)
    return sentence_without_punctuation


def find_correct_words_by_edit_distance(words_separated_by_space):
    # Lista per memorizzare le parole corrette
    correct_words = []

    for word in words_separated_by_space:
        found = False
        min_distance = float('inf')
        closest_word = None
        for token in app.tokenizer.get_vocab():

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

        for i in range(len(app.corpora_book_names)):
            tfidf_value = tfidf_matrix.toarray()[i][vectorizer.vocabulary_[token_key]]
            if(tfidf_value > 0.04):
                relevant_books.append(app.corpora_book_names[i])
    return list(set(relevant_books))

def generate_answer(books, question):
    if not books:
        return 'Nessuna risposta'
    min_score = 0
    context = ''
    best_answer = ''
    for book in books:
        file_path = os.path.join(corpora_dir, book)
        # Leggi il contenuto del file di testo
        with open(file_path, 'r', encoding='utf-8') as file:
            context += ' ' + file.read()

    qa_answer = qa_model(question = question, context = context)
    score = qa_answer['score']
    print(qa_answer)
    if score > min_score:
        best_answer = qa_answer

    return best_answer

@app.route('/')
def index():
    return render_template('index.html', sentence_results=[])
    
    
@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        sentence_results = []

        user_input = request.form['user_input']

        # Esegui la segmentazione delle frasi
        sentences = sent_tokenize(user_input)

        # Stampa le frasi
        for sentence in sentences:
            sentence_without_punctuation = remove_punctuation(sentence)

            encoded_text = app.tokenizer.encode(sentence_without_punctuation)
            tokenized_text = encoded_text.tokens

            words_separated_by_space = sentence_without_punctuation.split()
            # Trova le parole corrette utilizzando la distanza di edit
            correct_words = find_correct_words_by_edit_distance(words_separated_by_space)
            correct_words_text = ' '.join(correct_words)

            encoded_correct_text = app.tokenizer.encode(correct_words_text)
            # Restituisci i token corretti
            tokenized_correct_text = encoded_correct_text.tokens
            # Identificazione stopwords
            stopwords_removed = [word for word in tokenized_correct_text if word.lower() in stop_words]
            # Rimozione delle stopwords
            cleaned_text_no_stopwords = [word for word in tokenized_correct_text if word.lower() not in stop_words]
            # Lemmatizza il testo
            lemmatized_text = [lemmatizer.lemmatize(word) for word in cleaned_text_no_stopwords]

            relevant_books = identify_relevant_books(lemmatized_text, vectorizer, tfidf_matrix)

            print(relevant_books)

            generate_answer(relevant_books,sentence)

            result = Sentence(sentence=sentence_without_punctuation,
                              tokens=tokenized_text,
                              corrected_tokens=tokenized_correct_text,
                              tokens_no_stopwords=cleaned_text_no_stopwords,
                              lemmatized_text=lemmatized_text,
                              stopwords_removed=stopwords_removed)
            
            # Aggiungi il risultato alla lista dei risultati delle frasi
            sentence_results.append(result)

            # Converti manualmente gli oggetti Sentence in dizionari
            sentence_results_dict = [result.to_dict() for result in sentence_results]

        return render_template('index.html', sentence_results=sentence_results_dict)


if __name__ == '__main__':
    app.run(debug=True)
