import json
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

if __name__ == '__main__':
    # chrome driver used by selenium
    browser = webdriver.Chrome(ChromeDriverManager().install())

    # resulting dataset after pre-processing
    imdb_data = {}

    # contains a list of IMdB urls
    with open('movie_urls.txt', 'r') as f:
        urls = f.readlines()
    
    # cleans the urls list to remove Line Break characters
    urls = [i.rstrip() for i in urls]

    # iterates over each url in the list to retrieve movie reviews
    for url in urls:
        browser.get(url)
        while True:
            try:
                # automates the process of load more button to obtain all the reviews
                button = browser.find_element_by_xpath('//*[@id="load-more-trigger"]')
                button.click()
                time.sleep(5)
            # the loop breaks when "load-more" button is not present
            except Exception as e:
                break

        page_source = browser.page_source
        
        # using BeautifulSoup to parse the HTML page
        soup = BeautifulSoup(page_source, "html.parser")
        
        # initializing value of the movie name
        movie_name = ""
        
        # searches the HTML page to find related movie name class
        for item in soup.find_all(class_="subpage_title_block"):
            movie_name = str(item.find(class_="parent").text).rstrip('\r\n').lstrip().replace("\n", "").replace(" ", "")
            
            # key contains the movie name
            imdb_data[movie_name] = []

        # searches the HTML page to find reviews related movie name
        for item in soup.find_all(class_="review-container"):
            title_review = {}
            review_title = item.find(class_="title").text.rstrip()
            review = item.find(class_="text").text.rstrip().replace("\n", " ")
            title_review['title'] = review_title
            title_review['review'] = review
            
            # creates a list of reviews corresponding to the movie name
            imdb_data[movie_name].append(title_review)
        
        # create a JSON file to store the values
        with open('movie_reviews.json', 'w', encoding="utf-8") as f:
            json.dump(imdb_data, f, indent=2, ensure_ascii=False)
    
    # closes Google Chrome after processing has been completed
    browser.close()
