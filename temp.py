# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn import preprocessing
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split, KFold
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import matplotlib
from sklearn.utils import shuffle
from sklearn.utils import resample
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
import streamlit as st 

pickle_classifier = open("classifier.pkl","rb")

classifier=pickle.load(pickle_classifier)

def preprocessing(text):
    stemmer = WordNetLemmatizer()
    document = re.sub(r'\W', ' ', str(text))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    return document

def welcome():
    return "Welcome All"


def predict_note_authentication(sentence):
    
    w1 = preprocessing(sentence)
    
    list1 = []
    list1.append(w1)
    
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
    tfidf = transformer.fit_transform(loaded_vec.fit_transform(list1)).toarray() 
   
    prediction=classifier.predict(tfidf)
    
    if(prediction==0):
        return "a not Greeting"
    
    elif (prediction==1):
        return "a Greeting"



def main():
    st.title("Greetings!")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Automated Greeting Checking App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    sentence = st.text_input("Sentence","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(sentence)
    st.success('Your sentence is {}'.format(result))

if __name__=='__main__':
    main()