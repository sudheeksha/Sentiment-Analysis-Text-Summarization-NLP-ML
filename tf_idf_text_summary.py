import json
import math
import re
import textwrap

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

# threshold value
# threshold value is directly proportional to the length of the summary
THRESHOLD_VALUE = 1.8

# text wrapper to wrap lines and indentation to each line
wrapper = textwrap.TextWrapper(initial_indent='\t', subsequent_indent='\t', width=100)


def eliminate_url_emoji(string):
    """
    function to eliminate non ascii characters and urls in text
    :param string: represents one review
    :return: a string
    """
    input_text = string.encode('ascii', 'ignore').decode('ascii')
    input_text = re.sub(r'http\S+', '', input_text)
    return input_text+"\n"+"\n"


def eliminate_stopwords_stemming(text):
    """
    function eliminates stop words and returns a frequency matrix,
    the frequency matrix is a dictionary that contains key, value pairs
    in which the key represents a sentence and the value represents the count
    or frequency of occurrence.
    :param text: a collection of reviews
    :return: a word frequency matrix
    """
    # initializing nltk defined set of stop words for english
    stop_words = set(stopwords.words("english"))

    # reducing words to their root form using stemming
    root = PorterStemmer()

    # creating dictionary for the word frequency table
    word_frequency = {}

    for sentence in text:
        freq = {}
        # splitting strings based on space and punctuation
        words = word_tokenize(sentence)
        for word in words:
            # reducing words to their root form using stemming
            word = root.stem(word)
            # removing stop words from consideration
            if word in stop_words:
                continue
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1
        word_frequency[sentence[:15]] = freq

    return word_frequency


def _idf_helper(freq):
    """
    helper function to construct inverse-document-frequency matrix
    :param freq: a word frequency matrix
    :return: no of documents containing word
    """
    words_in_document = {}
    for sentence, f_matrix in freq.items():
        for word, count in f_matrix.items():
            if word in words_in_document:
                words_in_document[word] += 1
            else:
                words_in_document[word] = 1
    return words_in_document


def summary_of_reviews(tf_idf_matrix, reviews, res_file):
    """
    generates a summary of reviews by computing a score for each review,
    score of a sentence = (term-frequency-value/length of sentence)
    average score is computed by adding all the scores and dividing it by the
    number of keys (associated with scores), a sentence is selected for text summarization
    if the sentence score is more than the product average score and threshold value,
    the threshold value is pre-defined, changing this value results in different lengths of
    a summary
    :param tf_idf_matrix: computed tf-idf dictionary
    :param reviews: original reviews are processed
    :param res_file: result file
    :return: a summary of reviews is generated
    """
    summary = ""
    scores = {}
    res_file.write("computing scores\n\n")
    for sentence, frequency in tf_idf_matrix.items():
        # initialize total score for the sentence to 0
        total_score = 0
        for word, score in frequency.items():
            total_score += score
        scores[sentence] = total_score/len(frequency)
    res_file.write("scores are:\n\n")
    # computing average scores
    sum_scores = sum(scores.values())
    wrapped = wrapper.fill(str(sum_scores))
    res_file.write(wrapped + '\n')
    average_score = sum_scores/len(scores)
    wrapped = wrapper.fill(str(average_score))
    res_file.write(wrapped + '\n')

    # generating a summary using a threshold value
    # based on testing, it was found that a higher threshold value generates a shorter summary
    res_file.write("generating summary for threshold value: \t" + str(THRESHOLD_VALUE) + "\n\n")
    for review in reviews:
        if review[:15] in scores and scores[review[:15]] >= THRESHOLD_VALUE * average_score:
            summary += " " + review
    res_file.write("Summary generated is:\n\n")
    wrapped = wrapper.fill(summary)
    res_file.write(wrapped + '\n')
    return summary


if __name__ == '__main__':
    with open('results.txt', 'w') as results_file:
        results_file.write("TF-IDF results file\n\n")
        # using a sampled dataset of movie reviews
        with open('movie_sampled_data.json', 'r') as f:
            results_file.write("Loading movie dataset\n\n")
            movie_file = json.load(f)
        for movie_name in movie_file:
            results_file.write("Generating results for:\t" + movie_name+"\n\n")
            # reinitializing variable for each movie
            all_reviews = ""
            for j in movie_file[movie_name]:
                # appending partially cleaned review to variable
                all_reviews += eliminate_url_emoji(j['review'])
            # convert sentences in to tokens
            sentences = sent_tokenize(all_reviews)
            # creating a dictionary for the word frequency table
            freq_of_words = eliminate_stopwords_stemming(sentences)

            # computing term frequency
            # term frequency (tf) = (no of times term appears in a document/total number of terms in document)
            # initializing term frequency matrix
            term_frequency_matrix = {}
            # st - sentence, fq - frequency
            results_file.write("computing term frequency\n\n")
            for sentence, frequency in freq_of_words.items():
                term_frequency = {}
                total_terms_in_document = len(frequency)
                for word, terms_in_document in frequency.items():
                    term_frequency[word] = terms_in_document / total_terms_in_document
                term_frequency_matrix[sentence] = term_frequency

            words_in_documents = _idf_helper(freq_of_words)
            results_file.write("cleaning the data...\n\n")
            # computing inverse document frequency
            # inverse document frequency (idf) = log10(total no of documents/number of documents containing the term)
            # initializing inverse document frequency
            results_file.write("computing inverse document frequency\n\n")
            inverse_document_frequency_matrix = {}
            for sentence, freq_matrix in freq_of_words.items():
                inverse_doc_freq = {}
                for word in freq_matrix.keys():
                    inverse_doc_freq[word] = math.log(len(sentences) / float(words_in_documents[word]), 10)
                inverse_document_frequency_matrix[sentence] = inverse_doc_freq

            # computing tf-idf matrix
            # function to compute tf-idf, tf-idf describes how important a word is in a collection of documents
            # high f-idf value is proportional to the number of times a word appears in a document.
            term_freq_inverse_doc_freq_matrix = {}
            results_file.write("computing term frequency * inverse document frequency\n\n")
            for (s1, f1), (s2, f2) in zip(term_frequency_matrix.items(), inverse_document_frequency_matrix.items()):
                tf_idf = {}
                for (w1, v1), (w2, v2) in zip(f1.items(), f2.items()):
                    tf_idf[w1] = float(v1 * v2)
                term_freq_inverse_doc_freq_matrix[s1] = tf_idf

            # generate summary
            summary_of_movie_reviews = summary_of_reviews(term_freq_inverse_doc_freq_matrix, sentences, results_file)
            print(summary_of_movie_reviews)
            break
