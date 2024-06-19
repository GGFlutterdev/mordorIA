import xlsxwriter

def generateExcel(result):
    workbook = xlsxwriter.Workbook("TokenizationResults.xlsx")
    worksheet = workbook.add_worksheet("firstSheet")
    #Salva i risultati nel file excel
    #Fornisce i nomi alle colonne
    worksheet.write(0, 0, "Tokens")
    worksheet.write(0, 1, "Corrected Tokens")
    worksheet.write(0, 2, "Stopwords Removed")  
    worksheet.write(0, 3, "Tokens without stopwords")
    worksheet.write(0, 4, "Lemmatized text")
    worksheet.write(0, 5, "Relevant books")

    #Salva i valori nelle colonne
    for index, token_text in enumerate(result.tokens):
        worksheet.write(index+1, 0, str(token_text))

    for index, correct_token_text in enumerate(result.corrected_tokens):
        worksheet.write(index+1, 1, str(correct_token_text))

    for index, removed_words in enumerate(result.stopwords_removed):
        worksheet.write(index+1, 2, str(removed_words))

    for index, clean_text in enumerate(result.tokens_no_stopwords):
        worksheet.write(index+1, 3, str(clean_text))

    for index, lemma_text in enumerate(result.lemmatized_text):
        worksheet.write(index+1, 4, str(lemma_text))

    for index, relevant_book in enumerate(result.relevant_books):
        worksheet.write(index+1, 5, str(relevant_book))
    workbook.close()