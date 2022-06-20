from flask import Flask, render_template, request, jsonify
import nltk
import nltk
import pandas as pd
import numpy as np
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from googlesearch import search
from tqdm import tqdm
#import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
#export CUDA_VISIBLE_DEVICES=""
def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext


def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned


def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

import pickle
import newspaper
from newspaper import Article

def get_data(url):
  article=Article(url)
  try:
    article.download()
  except Exception as e:
    return False,'',[],''
  try:
    article.parse()
    article.nlp()
    title=article.title
    keywords=article.keywords
    summary=article.summary
    if(len(title)<=5 or len(keywords)==0 or len(summary)<=5):
      return False,'',[],''
    else:
      return True,title,keywords,summary
  except Exception as e:
      return False,'',[],''
  print(e)
#from sentence_transformers import SentenceTransformer,util

app = Flask(__name__)
ps = PorterStemmer()

model = pickle.load(open('model.pkl', 'rb'))
tfidfvect = pickle.load(open('tfidfvect.pkl', 'rb'))
#cosmodel = pickle.load(open('cossmodel.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

def predict(text):
    test_list = search(text, tld='co.in', num=10, stop=20, pause=2)
    test_list = list(test_list)
    useless_domain = (
    'https://en.wikipedia', 'https://www.linkedin', 'https://in.linkedin', 'https://expertportals.com',
    'https://academictree.org', 'https://facebook.com', 'https://youtube.com', 'https://m.facebook.com',
    'https://www.facebook.com')

    results = list(filter(lambda x: not x.startswith(useless_domain), test_list))
    title_lt = []
    keyword_lt = []
    summary_lt = []
    for url in tqdm(results):
        reachable, title, keywords, summary = get_data(url)
        if (reachable):
            title_lt.append(title)
            keyword_lt.append(keywords)
            summary_lt.append(summary)
            test_df = pd.DataFrame({"summary": summary_lt, "title": title_lt})
            test_df['summary'].str.lower()
            test_df['summary'].apply(cleanHtml)
            test_df['summary'].apply(cleanPunc)
            test_df['summary'].apply(keepAlpha)
            test_df['summary'].apply(removeStopWords)
    vectorized_test = tfidfvect.transform(test_df['summary'])

    test_df['Random_prob'] = model.predict_proba(vectorized_test)[:,0]
    Av_random=np.average(test_df['Random_prob'])
    Av_random

    #embedding1 = cosmodel.encode(summary_lt, convert_to_tensor=True)
    #embedding2 = cosmodel.encode(text, convert_to_tensor=True)
    #cosine_scores = util.cos_sim(embedding1, embedding2)

    #def Average(cosine_scores):
        #return sum(cosine_scores) / len(cosine_scores)

    #average = Average(cosine_scores)
    #average


    prediction = 'Prediction of this news 📰 is REAL 🧐 ' if ((Av_random) >=0.5) else 'Prediction for this  news 📰 is FAKE 🧐'
    return prediction


@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    prediction = predict(text)
    return render_template('index.html', text=text, result=prediction)


@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)

if __name__ == "__main__":
    app.run()
