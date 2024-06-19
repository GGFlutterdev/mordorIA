from nltk.tokenize import sent_tokenize
from flask import current_app
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from concurrent.futures import ThreadPoolExecutor
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer, scoring
import torch
import os
import re

stop_words = set(stopwords.words('english'))
special_chars_regex = re.compile(r'[^a-zA-Z0-9\s]')

# Inizializza il modello BART
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def sentenceExtractionFromRelevantBooks(relevant_books, question_tokens, question_most_relevant_tokens):
    if len(relevant_books) == 0:
        return ["No relevant books found. Make a more detailed research"]
    unique_sentences = set()
    # Assicurati di essere nel contesto dell'applicazione
    with current_app.app_context():
        tokenizer = current_app.tokenizer

        with ThreadPoolExecutor() as executor:
            # Passa la funzione e i suoi parametri ai thread
            futures = [executor.submit(sentrenceExtractionFromSingleBook, book, tokenizer, question_tokens, question_most_relevant_tokens) for book in relevant_books]

            # Raccogli i risultati man mano che vengono completati
            for future in futures:
                sentences = future.result()
                for sentence in sentences:
                    if sentence not in unique_sentences:
                        unique_sentences.add(sentence)

    # Classifica le frasi estratte in base alla similarità con i token
    ranked_sentences = rank_sentences(question_tokens, list(unique_sentences))
    if rank_sentences:
        return ranked_sentences[:10]
    else:
        return ["No relevant sentence found, sorry."]

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
        return []

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

def giveAnswer(ranked_sentences):
    if len(ranked_sentences) == 0:
        return ["Impossible to generate an answer, sorry. Try to formulate your question in a better and more precise way."]
    answer = generateAnswer(ranked_sentences)
    calculate_rouge(answer, ranked_sentences)
    return [answer]

def generateAnswer(sentences):
    # Combina le frasi in un unico paragrafo
    combined_sentences = ' '.join(sentences)
    combined_sentences_no_special_chars = re.sub(special_chars_regex, '', combined_sentences)
    
    # Tokenizza il testo con il tokenizer di Hugging Face
    inputs = tokenizer.encode(combined_sentences_no_special_chars, return_tensors='pt', max_length=1024, truncation=True)
    
    # Genera il riassunto
    summary_ids = model.generate(inputs, num_beams=4, min_length=30, max_length=200, early_stopping=True)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def calculate_rouge(answer, ranked_sentences):
    answer = re.sub(special_chars_regex, '', answer)
    ranked_sentences = [re.sub(special_chars_regex, '', sentence) for sentence in ranked_sentences]

    # Inizializza lo scorer per ROUGE-1, ROUGE-2 e ROUGE-3
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3'], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for reference in ranked_sentences:
        scores = scorer.score(answer, reference)
        aggregator.add_scores(scores)
    
    # Ottieni i risultati medi

    result = aggregator.aggregate()
    for key, value in result.items():
        print(f"{key}:")
        print(f"  Precision: {value.mid.precision:.4f}")
        print(f"  Recall:    {value.mid.recall:.4f}")
        print(f"  F1 score:  {value.mid.fmeasure:.4f}")
