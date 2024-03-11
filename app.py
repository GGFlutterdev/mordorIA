from flask import Flask, render_template, request
from tokenizers import ByteLevelBPETokenizer
from datasheet import datasheet_bp
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os

app = Flask(__name__)

app.register_blueprint(datasheet_bp)

# Inizializza il token learner BPE
app.tokenizer = ByteLevelBPETokenizer()

# Directory contenente i file di testo per l'addestramento del token learner
corpus_dir = './lotr_hp_corpus'
app.corpora = []

nltk.download('stopwords')

# Lista delle stopwords nella lingua inglese
stop_words = set(stopwords.words('english'))
# Scarica il WordNet corpus per la lemmatizzazione delle parole
nltk.download('wordnet')

# Inizializza il lemmatizzatore
lemmatizer = WordNetLemmatizer()

# Funzione per leggere i file di testo nel corpus e aggiornare il token learner
def train_token_learner(tokenizer, corpus_dir):
    app.corpora = []  # Lista per mantenere il testo di ciascun file
    # Itera attraverso i file di testo nel corpus
    for filename in os.listdir(corpus_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(corpus_dir, filename)
            # Leggi il contenuto del file di testo
            with open(file_path, 'r', encoding='utf-8') as file:
                corpus = file.read()
                app.corpora.append(corpus)  # Aggiungi il testo alla lista
    # Aggiorna il token learner con il testo dei file
    tokenizer.train_from_iterator(app.corpora)

# Addestra il token learner con il corpus di testo
train_token_learner(app.tokenizer, corpus_dir)

@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        user_input = request.form['user_input']
        # Tokenizza il testo utilizzando il token learner BPE
        encoded_text = app.tokenizer.encode(user_input)
        # Restituisci i token al template HTML
        tokenized_text = encoded_text.tokens
        # Rimozione spazi ed ultimo carattere
        cleaned_text = [token.replace('Ġ', '') for token in tokenized_text if token.replace('Ġ', '')]
        # Identificazione stopwords
        stopwords_removed = [word for word in cleaned_text if word.lower() in stop_words]
        # Rimozione delle stopwords
        cleaned_text_no_stopwords = [word for word in cleaned_text if word.lower() not in stop_words]
        # Lemmatizza il testo
        lemmatized_text = [lemmatizer.lemmatize(word) for word in cleaned_text_no_stopwords]

        return render_template('index.html',
                               tokens=cleaned_text,
                               tokens_no_stopwords = cleaned_text_no_stopwords,
                               lemmatized_text = lemmatized_text,
                               stopwords_removed = stopwords_removed
                               )

if __name__ == '__main__':
    app.run(debug=True)
