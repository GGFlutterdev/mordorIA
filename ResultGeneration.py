from nltk.tokenize import sent_tokenize
from flask import current_app
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from concurrent.futures import ThreadPoolExecutor
import os
import re

stop_words = set(stopwords.words('english'))
special_chars_regex = re.compile(r'[^a-zA-Z0-9\s]')

def sentenceExtractionFromRelevantBooks(relevant_books, question_tokens, question_most_relevant_tokens):
    unique_sentences = set()
    # Assicurati di essere nel contesto dell'applicazione
    with current_app.app_context():
        tokenizer = current_app.tokenizer

        with ThreadPoolExecutor() as executor:
            # Passa la funzione e i suoi parametri ai thread
            futures = [executor.submit(sentrenceExtractionFromSingleBook, book, tokenizer, question_tokens, question_most_relevant_tokens) for book in relevant_books]

            # Raccogli i risultati man mano che vengono completati
            # Raccogli i risultati man mano che vengono completati
            for future in futures:
                sentences = future.result()
                for sentence in sentences:
                    if sentence not in unique_sentences:
                        unique_sentences.add(sentence)

            # Classifica le frasi estratte in base alla similarità con i token
            ranked_sentences = rank_sentences(question_tokens, list(unique_sentences))
            
            # Restituisci le prime 10 frasi migliori se ce ne sono
            print(ranked_sentences[:10] if ranked_sentences else ["Nessuna frase rilevante trovata."])
            return ranked_sentences[:10] if ranked_sentences else ["Nessuna frase rilevante trovata."]

def sentrenceExtractionFromSingleBook(book, tokenizer, question_tokens, question_most_relevant_tokens):
    sentencesExtracted = []
    file_path = os.path.join('./corpora', book)
    # Leggi il contenuto del file di testo
    with open(file_path, 'r', encoding='utf-8') as file:
        bookContent = file.read()
        sentences = sent_tokenize(bookContent)

        for sentence in sentences:

            sentence_no_special_chars = re.sub(special_chars_regex, '', sentence)
            tokenized_sentence = tokenizer.tokenize(sentence_no_special_chars)

            for token in question_most_relevant_tokens:
                if token in tokenized_sentence and sentence not in sentencesExtracted:
                    sentencesExtracted.append(sentence)
                    break
    return sentencesExtracted

def rank_sentences(tokens, sentences):
    if not sentences:
        return ["Nessuna frase rilevante trovata."]

    # Inizializza il modello di trasformatori per embedding di frasi
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Trasforma i token in una stringa
    token_string = ' '.join(tokens)
    
    # Ottieni l'embedding della stringa di token e delle frasi
    question_embedding = model.encode(token_string, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    
    # Calcola la similarità coseno tra l'embedding della domanda e gli embedding delle frasi
    cosine_similarities = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0]

    # Ordina le frasi in base alla similarità coseno
    ranked_sentences = [sentence for _, sentence in sorted(zip(cosine_similarities, sentences), key=lambda x: x[0], reverse=True)]
    return ranked_sentences