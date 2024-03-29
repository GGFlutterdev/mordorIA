from flask import Blueprint, current_app, render_template
from collections import Counter
from math import log

datasheet_bp = Blueprint('datasheet_bp', __name__)

from collections import Counter
from math import log

def create_plot_data(tokenizer, corpora):
    # Inizializza le liste per i risultati complessivi
    all_tokens = []
    all_cleaned_tokens = []
    
    # Elabora ciascun testo nella lista di corpora
    for corpus in corpora:
        # Tokenizza il corpus di testo con il tokenizer
        encoded_corpus = tokenizer.encode(corpus)
        
        # Estrai i token dal corpus tokenizzato
        tokens = encoded_corpus.tokens
        all_tokens.extend(tokens)
        
        # Rimuovi il prefisso 'Ġ' dai token e aggiungi alla lista di token puliti
        cleaned_tokens = [token.replace('Ġ', '') for token in tokens]
        all_cleaned_tokens.extend(cleaned_tokens)
    
    # Calcola la frequenza delle parole per l'intero corpus
    word_freq = Counter(all_cleaned_tokens)

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
    corpora = current_app.corpora

    # Ottieni i dati del grafico
    plot_data = create_plot_data(tokenizer, corpora)
    

    # Restituisci il template HTML con i dati del grafico
    return render_template('datasheet.html', plot_data=plot_data)
