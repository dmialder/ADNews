import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist

nltk.download('punkt_tab')
nltk.download('stopwords')

def text_summarizer(text, num_sentences=3):
    # Text into sentences
    sentences = sent_tokenize(text)
    
    # Text into words
    words = word_tokenize(text.lower())
    
    # Removing stop words
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.casefold() not in stop_words]
    
    # Calculate word frequencies
    fdist = FreqDist(filtered_words)
    
    # Assign scores to sentences based on word frequencies
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in fdist:
                if i in sentence_scores:
                    sentence_scores[i] += fdist[word]
                else:
                    sentence_scores[i] = fdist[word]
    
    # Sort sentences by scores in descending order
    sorted_sentences = sorted(sentence_scores, key=lambda x: sentence_scores[x], reverse=True)
    
    # Select the top `num_sentences` sentences for the summary
    summary_sentences = sorted(sorted_sentences[:num_sentences])
    
    # Create the summary
    summary = ' '.join([sentences[i] for i in summary_sentences])
    
    return summary

# Example usage
text = """
Tech stocks were mixed Thursday afternoon with the Technology Select Sector SPDR Fund (XLK) decreasing 0.6% and the SPDR S&P Semiconductor ETF (XSD) climbing 3%.
The Philadelphia Semiconductor index rose 2.1%.
In corporate news, Adobe (ADBE) shares tumbled 6.5%, a day after fiscal 2024 revenue guidance disappointed investors.
MicroVision (MVIS) rose 1.9% after the company said it expects its 2023 revenue to be near the top end of its previous forecast of $6.5 million to $8 million.
Apple (AAPL) and Alphabet's (GOOG) Google were asked by the European Commission to provide information on risk-mitigation measures on their app stores, the EU's executive arm said Thursday. Apple fell 0.5%, and Alphabet dropped 1.9%.
The views and opinions expressed herein are the views and opinions of the author and do not necessarily reflect those of Nasdaq, Inc.s
"""

summary = text_summarizer(text)
print(summary)