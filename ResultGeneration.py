from nltk.tokenize import sent_tokenize
from flask import current_app
from UserInput import remove_g_prefix
import os

def sentenceExtractionFromRelevantBooks(relevant_books, tokens):
    sentencesExtracted = []
    for book in relevant_books:
        file_path = os.path.join('./corpora', book)
        # Leggi il contenuto del file di testo
        with open(file_path, 'r', encoding='utf-8') as file:
            bookContent = file.read()
            sentences = sent_tokenize(bookContent)
            for sentence in sentences:
                encoded_sentence = current_app.tokenizer.encode(sentence)
                tokenized_sentence = encoded_sentence.tokens
                tokenized_sentence = remove_g_prefix(tokenized_sentence)
                for token in tokens:
                    if token in sentence:
                        sentencesExtracted.append(sentence)
                        break

    # Classifica le frasi estratte in base alla similarità con i token
    ranked_sentences = rank_sentences(tokens, sentencesExtracted)
    
    # Restituisci le prime 5 frasi migliori se ce ne sono
    return ranked_sentences[:5] if ranked_sentences else ["Nessuna frase rilevante trovata."]

def rank_sentences(tokens, sentences):
    # Trasforma i token in una stringa
    token_string = ' '.join(tokens)
    
    # Usa la matrice TF-IDF esistente e calcola la similarità coseno
    question_tfidf = current_app.vectorizer.transform([token_string])
    sentence_tfidfs = current_app.vectorizer.transform(sentences)
    from sklearn.metrics.pairwise import cosine_similarity

    cosine_similarities = cosine_similarity(question_tfidf, sentence_tfidfs).flatten()
    ranked_sentences = [sentence for _, sentence in sorted(zip(cosine_similarities, sentences), reverse=True)]
    return ranked_sentences

def extractedSencencesEvaluation():
    return

def subsetOfRelevantSentences():
    return

def excelOfWorkflowReport():
    return