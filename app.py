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

import xlsxwriter

app = Flask(__name__)

app.register_blueprint(datasheet_bp)

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

# Inizializza il file excel
workbook = xlsxwriter.Workbook("TokenizationResults.xlsx")
worksheet = workbook.add_worksheet("firstSheet")

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

            relevant_books = identify_relevant_books(lemmatized_text, app.vectorizer, app.tfidf_matrix)

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

            #Salva i risultati nel file excel
            #Fornisce i nomi alle colonne
            worksheet.write(0, 0, "Tokens")
            worksheet.write(0, 1, "Corrected Tokens")
            worksheet.write(0, 2, "Stopwords Removed")  
            worksheet.write(0, 3, "Tokens without stopwords")
            worksheet.write(0, 4, "Lemmatized text")
            worksheet.write(0, 5, "Relevant books")

            #Salva i valori nelle colonne
            for index, token_text in enumerate(tokenized_text):
                worksheet.write(index+1, 0, str(token_text))

            for index, correct_token_text in enumerate(tokenized_correct_text):
                worksheet.write(index+1, 1, str(correct_token_text))

            for index, removed_words in enumerate(stopwords_removed):
                worksheet.write(index+1, 2, str(removed_words))

            for index, clean_text in enumerate(cleaned_text_no_stopwords):
                worksheet.write(index+1, 3, str(clean_text))

            for index, lemma_text in enumerate(lemmatized_text):
                worksheet.write(index+1, 4, str(lemma_text))

            for index, relevant_book in enumerate(relevant_books):
                worksheet.write(index+1, 5, str(relevant_book))
            workbook.close()

            # Converti manualmente gli oggetti Sentence in dizionari
            sentence_results_dict = [result.to_dict() for result in sentence_results]

        return render_template('index.html', sentence_results=sentence_results_dict)


if __name__ == '__main__':
    app.run(debug=True)
