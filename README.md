# CognitiveComputingProject

## Installation Required Libraries for Web Scraping

- Libraries: \
BeautifulSoup - pip3 install beautifulsoup4 \
Selenium - pip3 install -U selenium \
Web Driver Manager - pip3 install webdriver-manager 

## Required - Chrome Drivers:

https://chromedriver.chromium.org/downloads
          

## Movies Review URLs for Web Scraping:

https://www.imdb.com/title/tt6398184/reviews?ref_=tt_urv - Downtown Abbey (2019)

https://www.imdb.com/title/tt1025100/reviews?ref_=tt_urv - Gemini Man (2019)

https://www.imdb.com/title/tt5503686/reviews?ref_=tt_urv - Hustlers (2019)

https://www.imdb.com/title/tt7286456/reviews?ref_=tt_urv - Joker (2019)

https://www.imdb.com/title/tt7131622/reviews?ref_=tt_ov_rt - Once Upon a time in Hollywood (2019)


## Web Scraping

- Chrome web driver is OS dependent.
- Selenium is a framework for testing web apps, in this project selenium is used to automate the process of retreiving all reviews for a specified movie.
- movie_urls.txt contains links to the IMDb review page, if additional movies reviews are required the corresponding links can be added to the text file. This is processed by reviews_analysis.py
- reviews_analysis.py contains only one function, upon running this script the following steps take place
          - 


## Sentiment Analysis



## Text Summarization

### TextRank Algorithm 


### Term Frequency - Inverse Document Frequency

#### Running the code

- Required libraries are json, math, re, textwrap, nltk. 
- movie.txt is a text file which contains review of one movie
- 
- In default mode, tf_idf_text_summary.py reads data from movie.txt 
