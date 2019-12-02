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

Implementation Guidelines:
1. Ensure all the libraries below are installed
2. imbd_test.txt and imbd_train.txt are in the same directory as this python file.
3. Enter the file name of movie data file at prompt.
4. Ensure that movie data file is in same directory as this python file.
5. When prompted enter  '1' for logistic regression
                        '2' for svm
                        '3' for naive bayes
                        anything else for termination
"""
from pandas.io.json import json_normalize
import json
import re
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

stop_words = ['in', 'of', 'at', 'a', 'the']
replace_without_space = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
replace_with_space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|(\_)")
blank = ""
space = " "


def clean_review(review):
    """
    This function helps in cleaning the data of the movie to be predicted.
    :param review:
    :return:
    """

    # Excluding the html tags
    remove_html_tags = re.sub(r'<[^<>]+>', " ", review)
    # Converting numbers to "NUMBER"
    change_numbers = re.sub(r'[0-9]+', 'number', remove_html_tags)
    # Converting to lower case.
    lower_case_text = change_numbers.lower()
    # Replace puntuation marks with blank string
    clean_review = re.sub(r'[.;:!\'?,\"()\[\]_]',"", lower_case_text)
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


def clean_reviews(reviews):
    """
    This function cleans the IMBD data
    :param reviews:
    :return:
    """
    reviews = [replace_without_space.sub(blank, line.lower()) for line in reviews]
    reviews = [replace_with_space.sub(space, line) for line in reviews]

    return reviews


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


def logistic_regression(clean_words, train_reviews, test_reviews):
    '''
    This function performs logistic regression.
    :param cleanWords:
    :return:
    '''

    # Vectorize data

    cv = CountVectorizer(binary=True, ngram_range=(1, 2))
    cv.fit(train_reviews)
    X = cv.transform(train_reviews)
    X_test = cv.transform(test_reviews)

    target = [1 if i < 12500 else 0 for i in range(25000)]

    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, target, train_size=0.75
    # )

    # Fit data into the logistic regression function
    log_reg_model = LogisticRegression(C=0.05)
    log_reg_model.fit(X, target)

    # Make predictions.
    pos_reviews = 0
    neg_reviews = 0

    for i in range(len(clean_words)):
        # print('FOR REVIEW:     ' + cleanWords[i])
        if log_reg_model.predict(cv.transform([clean_words[i]])) == 1:
            pos_reviews += 1
            # print('RESULT:   1')

        if log_reg_model.predict(cv.transform([clean_words[i]])) == 0:
            neg_reviews += 1
            # print('RESULT:   0')

    print("Positive percentage with logistic regression:", pos_reviews/float(len(clean_words)))
    print("Negative percentage with logistic regression:", neg_reviews/float(len(clean_words)))

    print ("Accuracy of Logistic Regression: %s"% accuracy_score(target, log_reg_model.predict(X_test)))

    five_words(cv, log_reg_model)

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

    matrix = confusion_matrix(target, log_reg_model .predict(X_test))
    # confusion_matrix_plot(matrix)
    print(pd.DataFrame(matrix, columns=["Negatives", "Positives"], index=["Negatives", "Positives"]))


def svm(clean_words, train_reviews, test_reviews):
    # Vectorizer data
    cv = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
    cv.fit(train_reviews)
    X = cv.transform(train_reviews)
    X_test = cv.transform(test_reviews)

    target = [1 if i < 12500 else 0 for i in range(25000)]
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, target, train_size=0.75
    # )

    # Fit data into classifier
    svm_model = LinearSVC(C=0.01)
    svm_model.fit(X, target)

    print ("Accuracy of SVM: %s"% accuracy_score(target, svm_model.predict(X_test)))

    # Make predictions.
    pos_reviews = 0
    neg_reviews = 0

    for i in range(len(clean_words)):
        # print('FOR REVIEW:     ' + cleanWords[i])
        if svm_model.predict(cv.transform([clean_words[i]])) == 1:
            pos_reviews += 1
            # print('RESULT:   1')
        if svm_model.predict(cv.transform([clean_words[i]])) == 0:
            neg_reviews += 1
            # print('RESULT:   0')

    n = len(clean_words)
    n = n*1.0

    print("Positive percentage with svm:", (pos_reviews/n) * 100.0)
    print("Negative percentage with svm:", (neg_reviews/n) * 100.0)

    five_words(cv, svm_model)

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

    matrix = confusion_matrix(target, svm_model.predict(X_test))
    # confusion_matrix_plot(matrix)
    print(pd.DataFrame(matrix, columns=["Negatives", "Positives"], index=["Negatives", "Positives"]))


def naive_bayes(clean_words, train_reviews, test_reviews):
    # Vectorize data
    cv = CountVectorizer(binary=True, stop_words="english")
    cv.fit(train_reviews)
    X = cv.transform(train_reviews)
    X_test = cv.transform(test_reviews)

    target = [1 if i < 12500 else 0 for i in range(25000)]

    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, target, train_size=0.75
    # )

    # Fit data into classfier
    classifier = MultinomialNB()
    classifier.fit(X, target)

    # Make predictions
    pos_reviews = 0
    neg_reviews = 0
    n = len(clean_words)
    for i in range(len(clean_words)):
        # print('FOR REVIEW:     ' + cleanWords[i])
        if classifier.predict(cv.transform([clean_words[i]])) == 1:
            pos_reviews += 1
            # print('RESULT:   1')
        if classifier.predict(cv.transform([clean_words[i]])) == 0:
            neg_reviews += 1
            # print('RESULT:   0')

    print("Accuracy of Naive Bayes: %s"% accuracy_score(target, classifier.predict(X_test)))

    print("Positive percentage with Naive Bayes:", (pos_reviews/n) * 100.0)
    print("Negative percentage with Naive Bayes:", (neg_reviews/n) * 100.0)

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

    matrix = confusion_matrix(target, classifier.predict(X_test))
    # confusion_matrix_plot(matrix)
    print(pd.DataFrame(matrix, columns=["Negatives", "Positives"], index=["Negatives", "Positives"]))


def main():
    """
    Main program
    :return:
    """
    reviews_train = []
    for line in open('imbd_train.txt', 'r'):
        reviews_train.append(line.strip())

    reviews_test = []
    for line in open('imbd_test.txt', 'r'):
        reviews_test.append(line.strip())

    reviews_train_clean = clean_reviews(reviews_train)
    reviews_test_clean = clean_reviews(reviews_test)

    file_name = input('Enter the json file name :')
    with open(file_name+'.json') as f:
        d = json.load(f)
    df = json_normalize(d[file_name])
    cleanWords = []

    for i in range(df['review'].size):
        cleanWords.append(clean_review(df["title"][i])+ ' ' + clean_review(df["review"][i]))

    s = ''
    while s != 'stop':
        s = input(
            " Enter '1' for logistic regression\n '2' for svm\n '3' for naive bayes\n anything else for termination")
        if s == '1':
            logistic_regression(cleanWords, reviews_train_clean, reviews_test_clean)
        elif s == '2':
            svm(cleanWords, reviews_train_clean, reviews_test_clean)
        elif s == '3':
            naive_bayes(cleanWords, reviews_train_clean, reviews_test_clean)
        else:
            s = 'stop'

    # text_blob(cleanWords)


if __name__ == '__main__':
    main()
