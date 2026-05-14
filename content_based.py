import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf_matrix = None
        self.indices = None
        self.movies = None
        
    def fit(self, movies):
        """Build content-based model using movie genres"""
        self.movies = movies
        
        # Create TF-IDF matrix from genres
        tfidf = TfidfVectorizer(tokenizer=lambda x: x.split('|'), token_pattern=None)
        self.tfidf_matrix = tfidf.fit_transform(movies['genres'].fillna(''))
        
        # Create index mapping
        self.indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
        
        return self
    
    def recommend_similar_movies(self, movie_title, top_n=10):
        """Get top-N similar movies based on content"""
        if movie_title not in self.indices:
            # Try to find similar title
            matching_titles = [title for title in self.indices.index if movie_title.lower() in title.lower()]
            if matching_titles:
                movie_title = matching_titles[0]
            else:
                return pd.DataFrame()
        
        idx = self.indices[movie_title]
        
        # Calculate similarity scores
        sim_scores = list(enumerate(cosine_similarity(self.tfidf_matrix[idx:idx+1], self.tfidf_matrix)[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N (excluding the movie itself)
        sim_scores = sim_scores[1:top_n+1]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return recommendations
        recommendations = self.movies.iloc[movie_indices][['movieId', 'title', 'genres']].copy()
        recommendations['similarity_score'] = [i[1] for i in sim_scores]
        
        return recommendations