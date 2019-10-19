from urllib.request import urlopen
from bs4 import BeautifulSoup

if __name__ == '__main__':

    url = urlopen('http://www.imdb.com/title/tt0111161/reviews?ref_=tt_ov_rt').read()
    soup = BeautifulSoup(url, "html.parser")
    for item in soup.find_all(class_="review-container"):
        review_title = item.find(class_="title").text
        review = item.find(class_="text").text
        try:
            rating = item.find(class_="point-scale").previous_sibling.text
        except:
            rating = ""
        print("Title: {}\nReview: {}\nRating: {}\n".format(review_title, review, rating))
