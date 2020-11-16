# Automated-greeting-checking-application
Please watch the demo video of the model.

The goal is to identify the presence of greetings using Relational Strategies in Customer Service (RSiCS) Dataset. The dataset link https://nextit-public.s3-us-west-2.amazonaws.com/rsics.html?fbclid=IwAR0CktLQtuPBaZNk03odCKdrjN3LjYl_ouuFBbWvyj-yQ-BvzJ0v_n9w9xo.
The dataset labels are imbalanced so upsampling is performed. After upsampling the train and test data are processed separately by removing punctuations, special characters, stop words, converting to lower case and lemmatization. Vocabulary is created using vectorizer on training data. For classification Multinomial naive bayes is used which is a specialized version of naive bayes designed to handle text documents using word counts. 


The model is demostrated on streamlit.

To run the file: 

Download dataset

Download pickle files of feature and classifier model.

Download temp.py 

On console run the file : streamlit run temp.py

