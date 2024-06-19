from flask import Flask, render_template, request
from nltk.tokenize import RegexpTokenizer
from datasheet import datasheet_bp
from UserInput import userinput_bp, find_correct_words_by_edit_distance, identify_relevant_books, remove_punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from Sentence import Sentence
from ResultGeneration import sentenceExtractionFromRelevantBooks, giveAnswer
from GenerateExcel import generateExcel

import nltk
import re
import os

import xlsxwriter

app = Flask(__name__)

app.register_blueprint(datasheet_bp)

app.corpora = []  # Lista per mantenere il testo di ciascun file
app.corpora_book_names = []
# Directory contenente i file di testo per l'addestramento del token learner
corpora_dir = './corpora'
results_dir = './results'
app.tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
app.vocabulary = set()


# Lista delle stopwords nella lingua inglese
stop_words = set(stopwords.words('english'))

# Inizializza il lemmatizzatore
lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Inizializza il file excel
workbook = xlsxwriter.Workbook("TokenizationResults.xlsx")
worksheet = workbook.add_worksheet("firstSheet")

# Funzione per leggere i file di testo nel corpus e aggiornare il token learner
def train_token_learner():
    special_chars_regex = re.compile(r'[^a-zA-Z0-9\s]')
    # Itera attraverso i file di testo nel corpus
    for filename in os.listdir(corpora_dir):

        file_path = os.path.join(corpora_dir, filename)
        
        # Leggi il contenuto del file di testo
        with open(file_path, 'r', encoding='utf-8') as file:
            corpus = file.read()
            corpus_no_special_tokens = re.sub(special_chars_regex, '', corpus)

            tokens = app.tokenizer.tokenize(corpus)

            app.vocabulary.update(tokens)
            app.corpora.append(corpus)  # Aggiungi il testo alla lista
            app.corpora_book_names.append(filename)

train_token_learner()

app.vectorizer = TfidfVectorizer(stop_words='english')

app.tfidf_matrix = app.vectorizer.fit_transform(app.corpora)


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
            #Rimuovo la punteggiatura, non serve qui
            sentence_without_punctuation = remove_punctuation(sentence)

            #Tokenizzo
            tokenized_text = app.tokenizer.tokenize(sentence_without_punctuation)
            
            # Trova le parole corrette utilizzando la distanza di edit
            correct_text = find_correct_words_by_edit_distance(tokenized_text)

            # Identificazione stopwords
            stopwords_removed = [word for word in correct_text if word.lower() in stop_words]

            # Rimozione delle stopwords
            cleaned_text_no_stopwords = [word for word in correct_text if word.lower() not in stop_words]

            # Lemmatizza il testo
            lemmatized_text = [lemmatizer.lemmatize(word) for word in cleaned_text_no_stopwords]

            #Ottengo i libri rilevanti
            relevant_books = identify_relevant_books(lemmatized_text)

            result_sentences = sentenceExtractionFromRelevantBooks(relevant_books, correct_text, lemmatized_text)

            answer = giveAnswer(result_sentences)

            result = Sentence(
                sentence= sentence_without_punctuation,
                tokens= tokenized_text,
                corrected_tokens= correct_text,
                tokens_no_stopwords= cleaned_text_no_stopwords,
                lemmatized_text= lemmatized_text,
                stopwords_removed= stopwords_removed,
                relevant_books= relevant_books,
                result_sentences= result_sentences,
                answer= answer
            )
            
            # Aggiungi il risultato alla lista dei risultati delle frasi
            sentence_results.append(result)

            #Genero l'Excel
            generateExcel(result)

            # Converti manualmente gli oggetti Sentence in dizionari
            sentence_results_dict = [result.to_dict() for result in sentence_results]

        return render_template('index.html', sentence_results=sentence_results_dict)


if __name__ == '__main__':
    app.run(debug=True)
