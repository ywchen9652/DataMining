# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:57:47 2020

@author: YIWUN CHEN
"""

#1)	Read in the data into a DataFrame
import pandas as pd
import os
def data2df (path, label):
    file, text = [], []
    for f in os.listdir(path):
        file.append(f)
        fhr = open(path+f, 'r', encoding='utf-8', errors='ignore') 
        t = fhr.read()
        text.append(t)
        fhr.close()
    return(pd.DataFrame({'file': file, 'text': text, 'class':label}))

dfneg = data2df('HealthProNonPro/HealthProNonPro/NonPro/', 0) 
dfpos = data2df('HealthProNonPro/HealthProNonPro/Pro/', 1) 

df = pd.concat([dfpos, dfneg], axis=0)
df.sample(frac=0.002)

#2)	Setup the data for Training/Testing. Use 20% for testing.
X, y = df['text'], df['class']

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=1)

Xtrain = Xtrain.copy()
Xtest = Xtest.copy()
ytrain = ytrain.copy()
ytest = ytest.copy()


#3)	Use Spacy to preprocess the data. Explore and pick appropriate preprocessing steps.


def custom_tokenizer(doc):

    # clean up text
    tokens = [token.lemma_.lower() # lemmatize and lower-case 
                        for token in doc 
                               if (
                                    len(token) >= 2 and # only preserve tokens that are 2 or more characters long
                                    #token.pos_ in ['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV'] and # only preserve specific pos
                                    #token.text in nlp.vocab and # check if token in vocab
                                    #token.is_alpha and # only preserve tokens that are fully alpha (not numeric or alpha-numeric)
                                    #not token.is_digit and # get rid of tokens that are fully numeric
                                    not token.is_punct and # get rid of tokens that are punctuations
                                    not token.is_space and # get rid of tokens that are spaces
                                    not token.is_stop # get rid of tokens that are stop words
                                )
                   ]

    # return cleaned-up text
    return ' '.join(tokens)

import spacy
nlp = spacy.load("en_core_web_md", disable=['parser', 'ner'])
nlp_Xtrain = nlp.pipe(Xtrain)
clean_Xtrain = [custom_tokenizer(doc) for doc in nlp_Xtrain]
Xtrain= clean_Xtrain
print(Xtrain)


nlp_xtest = nlp.pipe(Xtest)
clean_xtest = [custom_tokenizer(doc) for doc in nlp_xtest]
Xtest= clean_xtest
print(Xtest)

#4)	Setup a Pipeline with TfidfVectorizer and Naïve Bayes. 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

clf= Pipeline(steps=[
    ('tfidf',TfidfVectorizer(binary=False, sublinear_tf=True, use_idf=True, smooth_idf=True, 
    norm='l2',lowercase=False, stop_words='english',min_df=1, max_df=1.0, max_features=None, ngram_range=(1, 1))),
    ('nb',MultinomialNB( alpha=1.0, fit_prior=True, class_prior=None))
])

#5)	Do Grid Search with 4-fold Cross Validation to search for the best values for the following two hyper-parameters (and any additional hyper parameters you may want to tune):
#o	sublinear_tf in TfidfVectorizer 
#o	alpha in Naïve Bayes 
from sklearn.model_selection import GridSearchCV
param_grid = {
    'tfidf__sublinear_tf': [True,False], 
    'nb__alpha':[1,0.8,0.6,0.4,0.2]
    }
gscv = GridSearchCV(clf, param_grid, cv=4, return_train_score=False)
gscv.fit(Xtrain, ytrain)

#6)	Use the Best Estimator resulting from the Grid Search for Prediction/Evaluation. Print the following evaluation metrics:
#o	Accuracy score
#o	Confusion matrix
#o	Classification report
print(gscv.best_estimator_, "\n")
print('------------------------------------------')
print(gscv.best_params_, "\n")
print('------------------------------------------')
ypred=gscv.predict(Xtest)
from sklearn import metrics
print (metrics.accuracy_score(ytest, ypred))
print (metrics.confusion_matrix(ytest, ypred))
print (metrics.classification_report(ytest, ypred))

#7)	Extract the true negatives (TN), false positives (FP), false negatives (FN), and true positives (TP) using the following command. 
TN, FP, FN, TP = metrics.confusion_matrix(y_true=ytest, y_pred=ypred).ravel()
print('TN=',TN,'FP=',FP,'FN=',FN,'TP=',TP)
print('------------------------------------------')
#Then, using TN/FP/FN/TP write code to calculate the overall accuracy, the precision (for class 0 and class 1), the recall (for class 0 and class 1), and the f1-score (for class 0 and class 1). 
#These should match what you are seeing in the accuracy_score and classification_report you printed above.
overall_accuracy_0= (TP+TN)/(TP+TN+FP+FN)
precision_0=  TN/(TN+FN)
recall_0= TN/(TN+FP)
f1_score_0= 2*TN/((2*TN)+FP+FN)
print('For Class 0:')
print('Overall_Accuracy=', round(overall_accuracy_0,2))
print('Precision=',  round(precision_0,2))
print('Recall=', round(recall_0,2))
print('f1-score=',round(f1_score_0,2))
print('------------------------------------------')
overall_accuracy_1= (TP+TN)/(TP+TN+FP+FN)
precision_1=  TP/(TP+FP)
recall_1= TP/(TP+FN)
f1_score_1= 2*TP/((2*TP)+FP+FN)

print('For Class 1:')
print('Overall_Accuracy=', round(overall_accuracy_1,2))
print('Precision=',  round(precision_1,2))
print('Recall=', round(recall_1,2))
print('f1-score=',round(f1_score_1,2))
