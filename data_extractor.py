import json
import random

if __name__ == '__main__':

    # sample size initialization
    X = 50

    # dictionary stores reviews for all movies
    movie_data_samples = {}

    with open('movie_reviews.json', 'r') as f:
        movie_file = json.load(f)

    # looping through each movie in the list
    for movie_name in movie_file:

        # dictionary stores reviews for each movie, resets every loop
        movie = {}

        # shuffling the list of reviews to obtain a random sample
        random.shuffle(movie_file[movie_name])

        # retrieves 50 movie reviews
        list_of_reviews = movie_file[movie_name][:X]

        # creates json object for each movie individually
        movie[movie_name] = list_of_reviews

        movie_data_samples[movie_name] = list_of_reviews

        # storing the 50 reviews in a json file for each movie
        with open('MovieData/'+movie_name+'.json', 'w', encoding="utf-8") as f:
            json.dump(movie, f, indent=2, ensure_ascii=False)

        # storing reviews for all movies in one json file
        with open('movie_sampled_data.json', 'w', encoding="utf-8") as f:
            json.dump(movie_data_samples, f, indent=2, ensure_ascii=False)
