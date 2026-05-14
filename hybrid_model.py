import pandas as pd
import numpy as np

class HybridRecommender:
    def __init__(self, cf_model, content_model, alpha=0.6, beta=0.4):
        self.cf_model = cf_model
        self.content_model = content_model
        self.alpha = alpha  # Collaborative weight
        self.beta = beta    # Content weight
        
    def update_weights(self, alpha, beta):
        """Update hybrid weights"""
        self.alpha = alpha
        self.beta = beta
        
    def recommend(self, user_id, seed_movie, movies, ratings, top_n=10):
        """Get hybrid recommendations"""
        # Get content-based similar movies
        content_recs = self.content_model.recommend_similar_movies(seed_movie, top_n=top_n*2)
        
        if content_recs.empty:
            return pd.DataFrame()
        
        # Get collaborative predictions for these movies
        recommendations = []
        for _, row in content_recs.iterrows():
            movie_id = row['movieId']
            
            # Get collaborative score
            try:
                cf_score = self.cf_model.predict_rating(user_id, movie_id)
            except:
                cf_score = 3.0
            
            # Get content similarity score
            content_score = row.get('similarity_score', 0.5)
            
            # Calculate hybrid score (normalize to 0-1 range)
            hybrid_score = (self.alpha * cf_score / 5.0) + (self.beta * content_score)
            
            recommendations.append({
                'movieId': movie_id,
                'title': row['title'],
                'genres': row['genres'],
                'predicted_rating': cf_score,
                'hybrid_score': hybrid_score
            })
        
        # Sort by hybrid score and get top N
        recommendations = sorted(recommendations, key=lambda x: x['hybrid_score'], reverse=True)[:top_n]
        
        return pd.DataFrame(recommendations)
    
    def recommend_top_popular(self, ratings, movies, top_n=10):
        """Get top popular movies based on average rating and count"""
        # Calculate movie statistics
        movie_stats = ratings.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        
        movie_stats.columns = ['movieId', 'avg_rating', 'rating_count']
        
        # Filter movies with at least 5 ratings
        movie_stats = movie_stats[movie_stats['rating_count'] >= 5]
        
        # Sort by average rating
        movie_stats = movie_stats.sort_values('avg_rating', ascending=False)
        
        # Get top N
        top_movies = movie_stats.head(top_n)
        
        # Merge with movie info
        result = pd.merge(top_movies, movies[['movieId', 'title', 'genres']], on='movieId')
        
        return result[['movieId', 'title', 'genres', 'avg_rating', 'rating_count']]