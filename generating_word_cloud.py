from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import json
import re
stopwords = set(STOPWORDS)


def eliminate_url_emoji(input):
    input_text = input.encode('ascii', 'ignore').decode('ascii')
    input_text = re.sub(r'http\S+', '', input_text)
    return input_text


def show_word_cloud(data):
    """
    program to generate a word cloud
    :param data: collection of reviews
    :return:
    """
    word_cloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=1
    ).generate(str(data))
    plt.axis('off')
    plt.imshow(word_cloud)
    plt.show()


if __name__ == '__main__':
    with open('movie_reviews.json', 'r') as f:
        movie_file = json.load(f)
    for movie_name in movie_file:
        text = ""
        for j in movie_file[movie_name]:
            text += eliminate_url_emoji(j['review'])
        show_word_cloud(text)





