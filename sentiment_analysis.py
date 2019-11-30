from __future__ import division
from pandas.io.json import json_normalize
import json

import re
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC

from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

stop_words = ['in', 'of', 'at', 'a', 'the']


def reviewWords(review):
    data_train_Exclude_tags = re.sub(r'<[^<>]+>', " ", review)  # Excluding the html tags
    data_train_num = re.sub(r'[0-9]+', 'number', data_train_Exclude_tags)  # Converting numbers to "NUMBER"
    data_train_lower = data_train_num.lower()  # Converting to lower case.
    result = re.sub(r'[.;:!\'?,\"()\[\]]',"", data_train_lower)
    return result


def sentiment(review):
    '''
    Utility function to classify sentiment of passed tweet
    using textblob's sentiment method
    '''
    # create TextBlob object of passed tweet text
    analysis = TextBlob(review)
        # set sentiment
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'


def text_blob(cleanWords):
    p = 0
    n = 0
    for i in range(len(cleanWords)):
        if sentiment(cleanWords[i]) == 'positive':
            p+=1
        if sentiment(cleanWords[i]) == 'negative':
            n+=1

    print("Positive percentage: {} %".format(100*p/len(cleanWords)))
    print("Negative percentage: {} %".format(100*(n)/len(cleanWords)))


reviews_train = []
for line in open('full_train.txt', 'r'):
    reviews_train.append(line.strip())

reviews_test = []
for line in open('full_test.txt', 'r'):
    reviews_test.append(line.strip())

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
NO_SPACE = ""
SPACE = " "


def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]

    return reviews


reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)


def logistic_regression(cleanWords):

    cv = CountVectorizer(binary=True)
    cv.fit(reviews_train_clean)
    X = cv.transform(reviews_train_clean)
    X_test = cv.transform(reviews_test_clean)

    target = [1 if i < 12500 else 0 for i in range(25000)]

    X_train, X_val, y_train, y_val = train_test_split(
        X, target, train_size=0.75
    )

    final_model = LogisticRegression(C=0.05)
    final_model.fit(X, target)

    p = 0
    n= 0
    for i in range(len(cleanWords)):
        if final_model.predict(cv.transform([cleanWords[i]])) == 1:
            p+=1
        if final_model.predict(cv.transform([cleanWords[i]])) == 0:
            n+=1

    print("Positive percentage with logistic regression:", p/float(len(cleanWords)))
    print("Negative percentage with logistic regression:", n/float(len(cleanWords)))

    print ("Final Accuracy: %s"
       % accuracy_score(target, final_model.predict(X_test)))


def svm(cleanWords):
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
    ngram_vectorizer.fit(reviews_train_clean)
    X = ngram_vectorizer.transform(reviews_train_clean)
    X_test = ngram_vectorizer.transform(reviews_test_clean)

    target = [1 if i < 12500 else 0 for i in range(25000)]
    X_train, X_val, y_train, y_val = train_test_split(
        X, target, train_size=0.75
    )

    final = LinearSVC(C=0.01)
    final.fit(X, target)
    print("Final Accuracy: %s"
      % accuracy_score(target, final.predict(X_test)))

    x = 0
    y = 0
    for i in range(len(cleanWords)):
        if final.predict(ngram_vectorizer.transform([cleanWords[i]])) == 1:
            x=x+1
        if final.predict(ngram_vectorizer.transform([cleanWords[i]])) == 0:
            y=y+1

    n = len(cleanWords)
    n = n*1.0

    print("Positive percentage with svm:", (x/n) * 100.0)
    print("Negative percentage with svm:", (y/n) * 100.0)


def naive_bayes(cleanWords):
    cv = CountVectorizer(binary=True)
    cv.fit(reviews_train_clean)
    X = cv.transform(reviews_train_clean)
    X_test = cv.transform(reviews_test_clean)

    target = [1 if i < 12500 else 0 for i in range(25000)]

    X_train, X_val, y_train, y_val = train_test_split(
        X, target, train_size=0.75
    )

    classifier = BernoulliNB().fit(X, target)
    x = 0
    y = 0
    n = len(cleanWords)
    for i in range(len(cleanWords)):
        if classifier.predict(cv.transform([cleanWords[i]])) == 1:
            x=x+1
        if classifier.predict(cv.transform([cleanWords[i]])) == 0:
            y=y+1

    print("Final Accuracy: %s"
      % accuracy_score(target, classifier.predict(X_test)))

    print("Positive percentage with Naive regression:", (x/n) * 100.0)
    print("Negative percentage with Naive regression:", (y/n) * 100.0)


def main():
    with open('DowntonAbbey(2019).json') as f:
        d = json.load(f)
    df = json_normalize(d['DowntonAbbey(2019)'])
    cleanWords = []

    for i in range(df['review'].size):
        cleanWords.append(reviewWords(df["title"][i])+ ' ' + reviewWords(df["review"][i]))
    naive_bayes(cleanWords)
    # svm(cleanWords)
    # logistic_regression(cleanWords)
    # text_blob(cleanWords)


if __name__ == '__main__':
    main()
    
