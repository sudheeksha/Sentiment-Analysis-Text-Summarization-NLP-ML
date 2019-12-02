"""
@author: Sudheeksha Garg

This program performs sentiment analysis using four different methods.
1. Logistic Regression
2. Naive Bayes
3. Support Vector Machine
4. Text Blob

There are four main steps in sentiment analysis:
1. Clean data
2. Vectorize data
3. Fit data into the choice of classifier
4. Make predictions.
"""
from pandas.io.json import json_normalize
import json
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import LatentDirichletAllocation

stop_words = ['in', 'of', 'at', 'a', 'the']


def clean_review(review):
    """
    This function helps in cleaning the data. 
    :param review: 
    :return: 
    """
    remove_html_tags = re.sub(r'<[^<>]+>', " ", review)  # Excluding the html tags
    change_numbers = re.sub(r'[0-9]+', 'number', remove_html_tags)  # Converting numbers to "NUMBER"
    lower_case_text = change_numbers.lower()  # Converting to lower case.
    clean_review = re.sub(r'[.;:!\'?,\"()\[\]_]',"", lower_case_text) # Replace puntuation marks with blank string
    return clean_review


def sentiment(review):
    '''
    Using text blob to classify sentiment of passed review
    '''
    # create TextBlob object of passed tweet text
    analysis = TextBlob(review)
    
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
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|(\_)")
NO_SPACE = ""
SPACE = " "


def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]

    return reviews


reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)


def five_words(cv, classifier):
    '''
    This function is used to find five most discrimating words for a particukar classifier.
    :param cv: 
    :param classifier: 
    :return: 
    '''
    
    feature_to_coef = {
        word: coef for word, coef in zip(
        cv.get_feature_names(), classifier.coef_[0]
    )
    }
    for best_positive in sorted(
            feature_to_coef.items(),
            key=lambda x: x[1],
            reverse=True)[:5]:
        print(best_positive)

    for best_negative in sorted(
            feature_to_coef.items(),
            key=lambda x: x[1])[:5]:
        print(best_negative)


def logistic_regression(cleanWords):
    '''
    This function performs logistic regression.
    :param cleanWords: 
    :return: 
    '''
    
    # Vectorize data
    cv = CountVectorizer(binary=True, ngram_range=(1, 2))
    cv.fit(reviews_train_clean)
    X = cv.transform(reviews_train_clean)
    X_test = cv.transform(reviews_test_clean)

    target = [1 if i < 12500 else 0 for i in range(25000)]

    X_train, X_val, y_train, y_val = train_test_split(
        X, target, train_size=0.75
    )
    
    # Fit data into the logistic regression function
    final_model = LogisticRegression(C=0.05)
    final_model.fit(X, target)

    # Make predictions.
    p = 0
    n= 0
    for i in range(len(cleanWords)):
        # print('FOR REVIEW:     ' + cleanWords[i])
        if final_model.predict(cv.transform([cleanWords[i]])) == 1:
            p+=1
            # print('RESULT:   1')
        if final_model.predict(cv.transform([cleanWords[i]])) == 0:
            n+=1
            # print('RESULT:   0')

    print("Positive percentage with logistic regression:", p/float(len(cleanWords)))
    print("Negative percentage with logistic regression:", n/float(len(cleanWords)))

    print ("Final Accuracy: %s"
       % accuracy_score(target, final_model.predict(X_test)))

    five_words(cv, final_model)

    # ('excellent', 0.7892975380298372)
    # ('perfect', 0.6670913277006437)
    # ('great', 0.6482132507785857)
    # ('wonderful', 0.5526971675176603)
    # ('amazing', 0.5165111647987498)
    #
    # ('worst', -0.9411044562700391)
    # ('awful', -0.8661982311505837)
    # ('boring', -0.7855300843689661)
    # ('waste', -0.74877595169182)
    # ('bad', -0.7283026032014535)


def svm(cleanWords):
    # Vectorizer data 
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
    ngram_vectorizer.fit(reviews_train_clean)
    X = ngram_vectorizer.transform(reviews_train_clean)
    X_test = ngram_vectorizer.transform(reviews_test_clean)

    target = [1 if i < 12500 else 0 for i in range(25000)]
    X_train, X_val, y_train, y_val = train_test_split(
        X, target, train_size=0.75
    )
    
    # Fit data into classifier
    final = LinearSVC(C=0.01)
    final.fit(X, target)
    print("Final Accuracy: %s"
      % accuracy_score(target, final.predict(X_test)))

    # Make predictions.
    x = 0
    y = 0
    for i in range(len(cleanWords)):
        # print('FOR REVIEW:     ' + cleanWords[i])
        if final.predict(ngram_vectorizer.transform([cleanWords[i]])) == 1:
            x=x+1
            # print('RESULT:   1')
        if final.predict(ngram_vectorizer.transform([cleanWords[i]])) == 0:
            y=y+1
            # print('RESULT:   0')

    n = len(cleanWords)
    n = n*1.0

    print("Positive percentage with svm:", (x/n) * 100.0)
    print("Negative percentage with svm:", (y/n) * 100.0)

    five_words(ngram_vectorizer, final)

    # ('excellent', 0.2304771495877233)
    # ('perfect', 0.18507025746313568)
    # ('great', 0.17881802771036037)
    # ('wonderful', 0.16078926606398075)
    # ('amazing', 0.1522694980219216)
    # ('worst', -0.35958635587583593)
    # ('awful', -0.2554010177475397)
    # ('boring', -0.2404542414522294)
    # ('waste', -0.23777958311617217)
    # ('bad', -0.22229430620803803)


def naive_bayes(cleanWords):
    # Vectorize data
    cv = CountVectorizer(binary=True, stop_words="english")
    cv.fit(reviews_train_clean)
    X = cv.transform(reviews_train_clean)
    X_test = cv.transform(reviews_test_clean)

    target = [1 if i < 12500 else 0 for i in range(25000)]

    X_train, X_val, y_train, y_val = train_test_split(
        X, target, train_size=0.75
    )

    # Fit data into classfier
    classifier = MultinomialNB()
    classifier.fit(X, target)
    
    # Make predictions
    x = 0
    y = 0
    n = len(cleanWords)
    for i in range(len(cleanWords)):
        # print('FOR REVIEW:     ' + cleanWords[i])
        if classifier.predict(cv.transform([cleanWords[i]])) == 1:
            x=x+1
            # print('RESULT:   1')
        if classifier.predict(cv.transform([cleanWords[i]])) == 0:
            y=y+1
            # print('RESULT:   0')

    print("Final Accuracy: %s"
      % accuracy_score(target, classifier.predict(X_test)))

    print("Positive percentage with Naive Bayes:", (x/n) * 100.0)
    print("Negative percentage with Naive Bayes:", (y/n) * 100.0)

    five_words(cv, classifier)
    # ('film', -5.164849518281162)
    # ('movie', -5.168706390428836)
    # ('like', -5.430627072171609)
    # ('good', -5.553771301061147)
    # ('just', -5.588659804732064)
    # ('aaaaaaah', -14.020512948981287)
    # ('aaaaah', -14.020512948981287)
    # ('aaaahhhhhhh', -14.020512948981287)
    # ('aaaarrgh', -14.020512948981287)
    # ('aaah', -14.020512948981287)

def main():
    """
    Main program
    :return: 
    """
    with open('DowntonAbbey(2019).json') as f:
        d = json.load(f)
    df = json_normalize(d['DowntonAbbey(2019)'])
    cleanWords = []

    for i in range(df['review'].size):
        cleanWords.append(clean_review(df["title"][i])+ ' ' + clean_review(df["review"][i]))
    naive_bayes(cleanWords)
    svm(cleanWords)
    logistic_regression(cleanWords)
    text_blob(cleanWords)
    lda(cleanWords)

if __name__ == '__main__':
    main()
