class Sentence:
    def __init__(self, sentence, tokens, corrected_tokens, tokens_no_stopwords, lemmatized_text, stopwords_removed, relevant_books, result_sentences, bart_answer, gpt2_answer):
        self.sentence = sentence
        self.tokens = tokens
        self.corrected_tokens = corrected_tokens
        self.tokens_no_stopwords = tokens_no_stopwords
        self.lemmatized_text = lemmatized_text
        self.stopwords_removed = stopwords_removed
        self.relevant_books = relevant_books
        self.result_sentences = result_sentences
        self.bart_answer = bart_answer
        self.gpt2_answer = gpt2_answer

    def to_dict(self):
        return {
            "sentence": self.sentence,
            "tokens": self.tokens,
            "corrected_tokens": self.corrected_tokens,
            "tokens_no_stopwords": self.tokens_no_stopwords,
            "lemmatized_text": self.lemmatized_text,
            "stopwords_removed": self.stopwords_removed,
            "relevant_books": self.relevant_books,
            "result_sentences": self.result_sentences,
            "bart_answer": self.bart_answer,
            "gpt2_answer": self.gpt2_answer
        }
