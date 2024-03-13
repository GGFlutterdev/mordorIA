from flask import Flask, render_template, request
from tokenizers import Tokenizer
from datasheet import datasheet_bp
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from sentence import Sentence
import editdistance
import nltk
import re

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

def remove_punctuation(sentence):
    # Definisci un'espressione regolare per la punteggiatura di fine frase
    punctuation_pattern = re.compile(r'[.!?,;:]')
    # Rimuovi la punteggiatura di fine frase dalla frase
    sentence_without_punctuation = punctuation_pattern.sub('', sentence)
    return sentence_without_punctuation

def find_correct_words(words_separated_by_space):
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
            correct_words = find_correct_words(words_separated_by_space)
            correct_words_text = ' '.join(correct_words)

            encoded_correct_text = app.tokenizer.encode(correct_words_text)
            # Restituisci i token al template HTML
            tokenized_correct_text = encoded_correct_text.tokens
            # Identificazione stopwords
            stopwords_removed = [word for word in tokenized_correct_text if word.lower() in stop_words]
            # Rimozione delle stopwords
            cleaned_text_no_stopwords = [word for word in tokenized_correct_text if word.lower() not in stop_words]
            # Lemmatizza il testo
            lemmatized_text = [lemmatizer.lemmatize(word) for word in cleaned_text_no_stopwords]

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
