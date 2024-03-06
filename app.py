from flask import Flask, render_template, request
from tokenizers import ByteLevelBPETokenizer
from datasheet import datasheet_bp
import os

app = Flask(__name__)

app.register_blueprint(datasheet_bp)

# Inizializza il token learner BPE
app.tokenizer = ByteLevelBPETokenizer()

# Directory contenente i file di testo per l'addestramento del token learner
corpus_dir = './lotr_hp_corpus'
app.corpus = ''

# Funzione per leggere i file di testo nel corpus e aggiornare il token learner
def train_token_learner(tokenizer, corpus_dir):
    
    # Itera attraverso i file di testo nel corpus
    for filename in os.listdir(corpus_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(corpus_dir, filename)
            # Leggi il contenuto del file di testo
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                app.corpus += text
                # Aggiorna il token learner con il testo del file
    tokenizer.train_from_iterator([app.corpus])

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
        cleaned_text = [token.replace('Ġ', '') for token in tokenized_text]
        print(cleaned_text)
        return render_template('index.html', tokens=cleaned_text)

if __name__ == '__main__':
    app.run(debug=True)
