from flask import Blueprint, current_app, render_template
from collections import Counter
from math import log

datasheet_bp = Blueprint('datasheet_bp', __name__)

def create_plot_data(tokenizer, corpus):
    # Tokenizza il corpus di testo con il tokenizer
    encoded_corpus = tokenizer.encode(corpus)

    # Estrai i token dal corpus tokenizzato
    tokens = encoded_corpus.tokens
    cleaned_tokens = [token.replace('Ġ', '') for token in tokens]

    # Calcola la frequenza delle parole
    word_freq = Counter(cleaned_tokens)

    # Ordina le parole per frequenza e ottieni i loro rank
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    ranks = list(range(1, len(sorted_words) + 1))

    # Estrai le frequenze ordinate e calcola il logaritmo per entrambi gli assi (opzionale)
    frequencies = [freq for _, freq in sorted_words]
    log_ranks = [log(rank) for rank in ranks]
    log_frequencies = [log(freq) for freq in frequencies]
    
    # Estrai i nomi delle parole
    words = [word for word, _ in sorted_words]

    return {
        'log_ranks': log_ranks,
        'log_frequencies': log_frequencies,
        'word_names': words  # Aggiungi i nomi delle parole ai dati del grafico
    }

@datasheet_bp.route('/datasheet')
def datasheet():
    tokenizer = current_app.tokenizer
    corpus = current_app.corpus

    # Ottieni i dati del grafico
    plot_data = create_plot_data(tokenizer, corpus)
    

    # Restituisci il template HTML con i dati del grafico
    return render_template('datasheet.html', plot_data=plot_data)
