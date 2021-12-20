# python version - Python 3.9.6
import pandas as pd
import numpy as np
import nltk

nltk.download('wordnet')
import re
from bs4 import BeautifulSoup

# READING THE DATA
import csv

df = pd.read_csv(
    'C://Users//nikit//OneDrive//Desktop//Masters//Fall2021//ANLP//Homework//HW1//amazon_reviews_us_Kitchen_v1_00.tsv'
    , parse_dates=True, delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')

# KEEPING ONLY THE REVIEW AND RATINGS
df = df.drop(columns=['marketplace', 'customer_id', 'review_id', 'product_id',
                      'product_parent', 'product_title', 'product_category',
                      'helpful_votes', 'total_votes', 'vine', 'verified_purchase',
                      'review_headline', 'review_date'])
df.rename(columns={'star_rating': 'Ratings', 'review_body': 'Review'}, inplace=True)
columns_change = ['Review', 'Ratings']
df = df[columns_change]

# Calculating the statistics of the rating
# print(df.groupby(['Ratings']).count())

# Label making
# The reviews with rating 4,5 are labelled to be 1 and 1,2 are labelled as 0.
# Discard the reviews with rating 3'
df.loc[df['Ratings'] < 3, 'Label'] = 0
df.loc[df['Ratings'] > 3, 'Label'] = 1
df.loc[df['Ratings'] == 3, 'Label'] = "neutral"
pd.options.display.float_format = '{:,.0f}'.format

# Including the number of reviews for each of the three classes
stats = df.groupby(['Label']).Review.count()
print(" Statistics of three classes : Label 0 : {0}, Label 1 : {1}, Label neutral : {2}"
      .format(stats[0], stats[1], stats[2]))

# selecting 200000 labels from the dataset 100000 positive and 100000 negative
positive = df.query("Label == 1").sample(100000)
negative = df.query("Label == 0").sample(100000)

frames = [positive, negative]

result = pd.concat(frames)
# display(result)

df = result.sample(frac=1)

df.reset_index(inplace=True)

df = df.drop(columns='index')

# Printing the average character length of the Review before Data Cleaning
length_preCLeaning = df.Review.str.len().mean()
# print(length_preCLeaning)

# DATA CLEANING
# Convert the all reviews into the lower case.
df['Review'] = df['Review'].str.lower()
# Remove the HTML and URLs from the reviews

def clean_html(html):
    try:
        soup = BeautifulSoup(html, "html.parser").text
        return soup
    except:
        text = ''
        return text
import bs4

df['Review'] = df['Review'].apply(lambda x: clean_html(x))
df['Review'] = df['Review'].str.replace('<.*?>', ' ', regex=True)
# Remove non-alphabetical characters
# Did not remove the apostrophe since they will be removed while expanding the contraction
df['Review'] = df['Review'].str.replace("[^a-zA-Z0-9']", " ", regex=True)
# Remove the extra spaces between the words
df['Review'] = df['Review'].str.replace('\s+', ' ', regex=True)

# EXPANDING CONTRACTIONS
# pip install contractions==0.0.18
import contractions


def contractionfunction(s):
    expanded_words = [contractions.fix(s)]
    expanded_text = ' '.join(expanded_words)
    s = expanded_text
    return s


df['Review'] = df['Review'].apply(str)
df['Review'] = df['Review'].apply(lambda x: contractionfunction(x))

# Printing the average character length of the Review after Data Cleaning
length_afterCLeaning = df.Review.str.len().mean()
print("Average length of reviews before and after data preprocessing : {0} and {1}"
      .format(length_preCLeaning, length_afterCLeaning))

# Pre-processing of the data
# Removing stop words
from nltk.corpus import stopwords

nltk.download('stopwords')
stop = stopwords.words('english')
df['Review'] = df['Review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

# Performing lemmatization
from nltk.stem import WordNetLemmatizer

lemmatizer = nltk.stem.WordNetLemmatizer()


def pract_lem(sentence):
    sente = nltk.word_tokenize(sentence)
    lemmatized_string = ' '.join([lemmatizer.lemmatize(words) for words in sente])
    return lemmatized_string


import nltk

nltk.download('punkt')
df['Review'] = df['Review'].apply(str)
df['Review'] = df['Review'].apply(lambda x: pract_lem(x))

# Printing the average character length of the Review after Data Cleaning and pre-processing
length_after = df.Review.str.len().mean()
print("Average length of reviews before and after data preprocessing : {0} and {1}"
      .format(length_afterCLeaning, length_after))

# TF-IDF Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer

vector_data = TfidfVectorizer().fit(df['Review'])
vectorized_data = vector_data.transform(df['Review'])

# Splitting of the dataset in the ratio of 80:20
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(vectorized_data, df['Label'], random_state=50, test_size=0.2)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

# PERCEPTRON MODEL
from sklearn.linear_model import Perceptron

Perceptron_model = Perceptron(max_iter=40, eta0=0.1, random_state=0)
Perceptron_model.fit(X_train, y_train)
PPredictions_traindata = Perceptron_model.predict(X_train)
PPredictions_testdata = Perceptron_model.predict(X_test)
from sklearn import metrics

# Accuracy, Precision, Recall, F1 Score for training data
P_accuracy = metrics.accuracy_score(y_train, PPredictions_traindata)
P_precision = metrics.precision_score(y_train, PPredictions_traindata)
P_recall = metrics.recall_score(y_train, PPredictions_traindata)
P_f1Score = metrics.f1_score(y_train, PPredictions_traindata)
print("Accuracy, Precision, Recall, and f1-score for training and testing split for Perceptron")
print("For training split")
print(" Accuracy: {0} \n Precision : {1} \n Recall : {2} \n F1_Score : {3}".format(P_accuracy, P_precision, P_recall,
                                                                                   P_f1Score))
# Accuracy, Precision, Recall, F1 Score for testing data
P_accuracy1 = metrics.accuracy_score(y_test, PPredictions_testdata)
P_precision1 = metrics.precision_score(y_test, PPredictions_testdata)
P_recall1 = metrics.recall_score(y_test, PPredictions_testdata)
P_f1Score1 = metrics.f1_score(y_test, PPredictions_testdata)
print("For testing split")
print(" Accuracy: {0} \n Precision : {1} \n Recall : {2} \n F1_Score : {3}".format(P_accuracy1, P_precision1, P_recall1,
                                                                                   P_f1Score1))

# SVM MODEL
# from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

svm_model = Pipeline([('scaler', StandardScaler(with_mean=False)), ('Linear_svc', LinearSVC(C=1, dual=False))])
svm_model.fit(X_train, y_train)
SVMPredictions_traindata = svm_model.predict(X_train)
SVMPredictions_testdata = svm_model.predict(X_test)
# Accuracy, Precision, Recall, F1 Score for training data
SVM_accuracy = metrics.accuracy_score(y_train, SVMPredictions_traindata)
SVM_precision = metrics.precision_score(y_train, SVMPredictions_traindata)
SVM_recall = metrics.recall_score(y_train, SVMPredictions_traindata)
SVM_f1Score = metrics.f1_score(y_train, SVMPredictions_traindata)
print("Accuracy, Precision, Recall, and f1-score for training and testing split for SVM")
print("For training split")
print(" Accuracy: {0} \n Precision : {1} \n Recall : {2} \n F1_Score : {3}".format(SVM_accuracy, SVM_precision,
                                                                                   SVM_recall, SVM_f1Score))
# Accuracy, Precision, Recall, F1 Score for testing data
SVM_accuracy1 = metrics.accuracy_score(y_test, PPredictions_testdata)
SVM_precision1 = metrics.precision_score(y_test, PPredictions_testdata)
SVM_recall1 = metrics.recall_score(y_test, PPredictions_testdata)
SVM_f1Score1 = metrics.f1_score(y_test, PPredictions_testdata)
print("For testing split")
print(" Accuracy: {0} \n Precision : {1} \n Recall : {2} \n F1_Score : {3}".format(SVM_accuracy1, SVM_precision1,
                                                                                   SVM_recall1, SVM_f1Score1))

# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

LR_model = LogisticRegression(random_state=0, max_iter=500).fit(X_train, y_train)
LRPredictions_traindata = LR_model.predict(X_train)
LRPredictions_testdata = LR_model.predict(X_test)
# Accuracy, Precision, Recall, F1 Score for training data
LR_accuracy = metrics.accuracy_score(y_train, LRPredictions_traindata)
LR_precision = metrics.precision_score(y_train, LRPredictions_traindata)
LR_recall = metrics.recall_score(y_train, LRPredictions_traindata)
LR_f1Score = metrics.f1_score(y_train, LRPredictions_traindata)
print("Accuracy, Precision, Recall, and f1-score for training and testing split for Logistic Regression")
print("For training split")
print(" Accuracy: {0} \n Precision : {1} \n Recall : {2} \n F1_Score : {3}".format(LR_accuracy, LR_precision, LR_recall,
                                                                                   LR_f1Score))
# Accuracy, Precision, Recall, F1 Score for testing data
LR_accuracy1 = metrics.accuracy_score(y_test, LRPredictions_testdata)
LR_precision1 = metrics.precision_score(y_test, LRPredictions_testdata)
LR_recall1 = metrics.recall_score(y_test, LRPredictions_testdata)
LR_f1Score1 = metrics.f1_score(y_test, LRPredictions_testdata)
print("For testing split")
print(" Accuracy: {0} \n Precision : {1} \n Recall : {2} \n F1_Score : {3}".format(LR_accuracy1, LR_precision1,
                                                                                   LR_recall1, LR_f1Score1))

# MULTINOMIAL NAIVE BAYES
from sklearn.naive_bayes import MultinomialNB

MNB_model = MultinomialNB().fit(X_train, y_train)
MNBPredictions_traindata = MNB_model.predict(X_train)
MNBPredictions_testdata = MNB_model.predict(X_test)
# Accuracy, Precision, Recall, F1 Score for training data
MNB_accuracy = metrics.accuracy_score(y_train, MNBPredictions_traindata)
MNB_precision = metrics.precision_score(y_train, MNBPredictions_traindata)
MNB_recall = metrics.recall_score(y_train, MNBPredictions_traindata)
MNB_f1Score = metrics.f1_score(y_train, MNBPredictions_traindata)
print("Accuracy, Precision, Recall, and f1-score for training and testing split for Naive Bayes")
print("For training split")
print(" Accuracy: {0} \n Precision : {1} \n Recall : {2} \n F1_Score : {3}".format(MNB_accuracy, MNB_precision,
                                                                                   MNB_recall, MNB_f1Score))
# Accuracy, Precision, Recall, F1 Score for testing data
MNB_accuracy1 = metrics.accuracy_score(y_test, MNBPredictions_testdata)
MNB_precision1 = metrics.precision_score(y_test, MNBPredictions_testdata)
MNB_recall1 = metrics.recall_score(y_test, MNBPredictions_testdata)
MNB_f1Score1 = metrics.f1_score(y_test, MNBPredictions_testdata)
print("For testing split")
print(" Accuracy: {0} \n Precision : {1} \n Recall : {2} \n F1_Score : {3}".format(MNB_accuracy1, MNB_precision1,
                                                                                   MNB_recall1, MNB_f1Score1))
