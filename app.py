from tokenizers.pre_tokenizers import Whitespace
from flask import Flask, render_template, request
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from datasheet import datasheet_bp
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

app = Flask(__name__)

app.register_blueprint(datasheet_bp)

# Inizializza il token learner BPE
app.tokenizer = Tokenizer.from_file("./data/tokenizer.json")

# Directory contenente i file di testo per l'addestramento del token learner
corpus_dir = './corpora'
app.corpora = []

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Lista delle stopwords nella lingua inglese
stop_words = set(stopwords.words('english'))

# Inizializza il lemmatizzatore
lemmatizer = WordNetLemmatizer()

@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        user_input = request.form['user_input']

        encoded_text = app.tokenizer.encode(user_input)
        # Restituisci i token al template HTML
        tokenized_text = encoded_text.tokens
        # Identificazione stopwords
        stopwords_removed = [word for word in tokenized_text if word.lower() in stop_words]
        # Rimozione delle stopwords
        cleaned_text_no_stopwords = [word for word in tokenized_text if word.lower() not in stop_words]
        # Lemmatizza il testo
        lemmatized_text = [lemmatizer.lemmatize(word) for word in cleaned_text_no_stopwords]

        return render_template('index.html',
                               tokens=tokenized_text,
                               tokens_no_stopwords = cleaned_text_no_stopwords,
                               lemmatized_text = lemmatized_text,
                               stopwords_removed = stopwords_removed
                               )

if __name__ == '__main__':
    app.run(debug=True)
