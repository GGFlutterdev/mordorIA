from nltk.tokenize import sent_tokenize
from flask import current_app
from UserInput import remove_g_prefix
import os

def sentenceExtractionFromRelevantBooks(relevant_books, tokens) :
    sentecesExtracted = []
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
                    if (token in sentence):
                        sentecesExtracted.append(sentence)
                        break
    return sentecesExtracted

def extractedSencencesEvaluation():
    return

def subsetOfRelevantSentences():
    return

def excelOfWorkflowReport():
    return