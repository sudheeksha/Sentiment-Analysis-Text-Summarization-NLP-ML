import time
import json
from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

if __name__ == '__main__':
    browser = webdriver.Chrome(ChromeDriverManager().install())
    imdb_data = {}
    url = "https://www.imdb.com/title/tt0094118/reviews?ref_=tt_ov_rt"
    browser.get(url)
    while True:
        try:
            button = browser.find_element_by_xpath('//*[@id="load-more-trigger"]')
            button.click()
            time.sleep(5)
        except Exception as e:
            break

    page_source = browser.page_source
    soup = BeautifulSoup(page_source, "html.parser")

    movie_name = ""
    for item in soup.find_all(class_="subpage_title_block"):
        movie_name = str(item.find(class_="parent").text).rstrip('\r\n').lstrip().replace("\n", "").replace(" ", "")
        imdb_data[movie_name] = []

    for item in soup.find_all(class_="review-container"):
        title_review = {}
        review_title = item.find(class_="title").text.rstrip()
        review = item.find(class_="text").text.rstrip().replace("\n", " ")
        title_review['title'] = review_title
        title_review['review'] = review
        imdb_data[movie_name].append(title_review)

    with open('movie_reviews.json', 'w', encoding="utf-8") as f:
        json.dump(imdb_data, f, indent=2, ensure_ascii=False)







