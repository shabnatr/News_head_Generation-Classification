from flask import Flask,render_template,url_for,request

import pickle
import numpy as np
# data processing
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

# for machine leanring
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn_evaluation.plot import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
plt.rcParams['figure.figsize'] = (20.0, 10.0)

df = pd.read_csv('data/uci-news-aggregator.csv')
df.isnull().any()
df[df.isnull().any(1)]
df.isnull().sum()
df['PUBLISHER']   = df['PUBLISHER'].replace('Unknown', np.nan)

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

df = pd.read_csv('data/uci-news-aggregator.csv')
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['TITLE'])
y = df['CATEGORY']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 

Pkl_Filename = "NB_Model.pkl"  
with open(Pkl_Filename, 'rb') as file:  
    NBM = pickle.load(file)
NBM
Pkl_Filename = "DT_Model.pkl"  
with open(Pkl_Filename, 'rb') as file:  
    DTM = pickle.load(file)

DTM
Pkl_Filename = "SGD_Model.pkl"  
with open(Pkl_Filename, 'rb') as file:  
    SGD = pickle.load(file)

SGD
SGD.best_params_
SGD.best_estimator_.score(X_test, y_test)
tit = input("News headline:")
def title_to_category(title):
    
           categories = {'b' : 'Business', 
                  't' : 'Science and Technology', 
                  'e' : 'Entertainment', 
                  'm' : 'Health'}
    
           vectorize_text = vectorizer.transform([title])
           predicter = SGD.best_estimator_.predict(vectorize_text)
           return categories[predicter[0]]
print("category:")
val = title_to_category(tit)


app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def home():
	return render_template('index.html')

@app.route("/helo", methods=["GET"])
def helo():
    article = ""
    with open(os.path.join("C:/Users/user/Desktop/jupyter2/NewsSummarization-master/NewsSummarization-master/articles/entertainment_a/002.txt"),encoding="utf-8") as f:
								for line in f:
										article += str(line)
	
		
										df = pd.read_fwf('C:/Users/user/Desktop/jupyter2/NewsSummarization-master/NewsSummarization-master/articles/entertainment_a/002.txt')
    article = re.sub(r'\s+',' ',article)
    Pkl_Filename = "PModel.pkl" 
    with open(Pkl_Filename, 'rb') as file:  
       Pickled_LR_Model = pickle.load(file)

    summary=Pickled_LR_Model
    	        
    return render_template('home.html',article = article, summary = summary)


@app.route('/predict',methods=["GET","POST"])
def predict():  
    #if request.method == 'POST':
       

      
    return render_template('result1.html',tit = tit, val=val)

@app.route('/pred',methods=["GET","POST"])
def pred():
    return render_template('res.html')

if __name__ == '__main__':
    app.run(debug=True)