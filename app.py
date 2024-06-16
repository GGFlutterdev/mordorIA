from flask import Flask, render_template, request
from tokenizers import ByteLevelBPETokenizer
from datasheet import datasheet_bp
from UserInput import userinput_bp, find_correct_words_by_edit_distance, identify_relevant_books, remove_g_prefix, remove_punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from Sentence import Sentence
from transformers import pipeline
from ResultGeneration import sentenceExtractionFromRelevantBooks


import nltk

import os

app = Flask(__name__)

app.register_blueprint(datasheet_bp)
app.register_blueprint(userinput_bp)

app.corpora = []  # Lista per mantenere il testo di ciascun file
app.corpora_book_names = []
# Directory contenente i file di testo per l'addestramento del token learner
corpora_dir = './corpora'
results_dir = './results'
app.tokenizer = ByteLevelBPETokenizer()
qa_model = pipeline("question-answering")

# Lista delle stopwords nella lingua inglese
stop_words = set(stopwords.words('english'))

# Inizializza il lemmatizzatore
lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# Funzione per leggere i file di testo nel corpus e aggiornare il token learner
def train_token_learner(corpora_dir):
    
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
    app.tokenizer.train_from_iterator(app.corpora, vocab_size=100000,
     show_progress=True,
     special_tokens=["<s>","<pad>","</s>","<unk>","<mask>"])

# Addestra il token learner con il corpus di testo
train_token_learner(corpora_dir)

vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = vectorizer.fit_transform(app.corpora)

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
            tokenized_text = remove_g_prefix(tokenized_text)
            

            words_separated_by_space = sentence_without_punctuation.split()
            # Trova le parole corrette utilizzando la distanza di edit
            correct_words = find_correct_words_by_edit_distance(words_separated_by_space)
            correct_words_text = ' '.join(correct_words)

            

            encoded_correct_text = app.tokenizer.encode(correct_words_text)
            # Restituisci i token corretti
            tokenized_correct_text = encoded_correct_text.tokens
            tokenized_correct_text = remove_g_prefix(tokenized_correct_text)
            # Identificazione stopwords
            stopwords_removed = [word for word in tokenized_correct_text if word.lower() in stop_words]
            # Rimozione delle stopwords
            cleaned_text_no_stopwords = [word for word in tokenized_correct_text if word.lower() not in stop_words]
            # Lemmatizza il testo
            lemmatized_text = [lemmatizer.lemmatize(word) for word in cleaned_text_no_stopwords]

            relevant_books = identify_relevant_books(lemmatized_text, vectorizer, tfidf_matrix)

            print(sentenceExtractionFromRelevantBooks(relevant_books, lemmatized_text))

            result = Sentence(
                        sentence= sentence_without_punctuation,
                        tokens= tokenized_text,
                        corrected_tokens= tokenized_correct_text,
                        tokens_no_stopwords= cleaned_text_no_stopwords,
                        lemmatized_text= lemmatized_text,
                        stopwords_removed= stopwords_removed,
                        relevant_books= relevant_books
                    )
            
            # Aggiungi il risultato alla lista dei risultati delle frasi
            sentence_results.append(result)

            # Converti manualmente gli oggetti Sentence in dizionari
            sentence_results_dict = [result.to_dict() for result in sentence_results]

        return render_template('index.html', sentence_results=sentence_results_dict)


if __name__ == '__main__':
    app.run(debug=True)
