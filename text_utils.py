# text_utils.py
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
import numpy as np
import nltk

for res in ['stopwords', 'wordnet']:
    try:
        nltk.data.find(f'corpora/{res}')
    except LookupError:
        nltk.download(res)


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9 ]+', '', text)
    text = re.sub(r'(http|https|ftp|ssh)://\S+', '', text)
    text = BeautifulSoup(text, 'lxml').get_text()
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = " ".join(lemmatizer.lemmatize(word) for word in text.split())
    text = " ".join(text.split())
    return text

def avg_word2vec(doc, w2v_model):
    vectors = [w2v_model.wv[word] for word in doc if word in w2v_model.wv.key_to_index]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)


# import numpy as np
# # Your avg_word2vec function
# def avg_word2vec(doc, w2v_model):
#     vectors = [w2v_model.wv[word] for word in doc if word in w2v_model.wv.key_to_index]
#     if vectors:
#         return np.mean(vectors, axis=0)
#     else:
#         return np.zeros(w2v_model.vector_size)