import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class DataPreprocessor:
    def __init__(self):
        self.movies = None
        self.ratings = None
        
    def load_movies(self, movies_path='movies.csv'):
        self.movies = pd.read_csv(movies_path)
        print(f"Loaded {len(self.movies)} movies")
        return self.movies
    
    def load_ratings(self, ratings_path='ratings.csv'):
        self.ratings = pd.read_csv(ratings_path)
        print(f"Loaded {len(self.ratings)} ratings")
        return self.ratings
    
    def clean_ratings(self, ratings):
        ratings = ratings.drop_duplicates(subset=['userId', 'movieId'])
        ratings = ratings[(ratings['rating'] >= 0.5) & (ratings['rating'] <= 5.0)]
        ratings = ratings.dropna()
        return ratings
    
    def build_processed_dataset(self):
        movies = self.load_movies()
        ratings = self.load_ratings()
        ratings = self.clean_ratings(ratings)
        merged = pd.merge(ratings, movies, on='movieId')
        train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)
        return merged, train_df, test_df
    
    def get_movie_catalog(self):
        if self.movies is None:
            self.load_movies()
        return self.movies
    
    def get_statistics(self):
        if self.movies is None:
            self.load_movies()
        if self.ratings is None:
            self.load_ratings()
        stats = {
            'num_movies': len(self.movies),
            'num_ratings': len(self.ratings),
            'num_users': self.ratings['userId'].nunique(),
            'avg_rating': self.ratings['rating'].mean(),
            'rating_density': (len(self.ratings) / (self.ratings['userId'].nunique() * len(self.movies))) * 100
        }
        return stats
      