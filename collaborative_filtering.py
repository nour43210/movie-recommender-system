import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import pickle
import os

class CollaborativeFilteringRecommender:
    """Collaborative Filtering using SVD (Singular Value Decomposition)"""
    
    def __init__(self, n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.svd = None
        self.user_matrix = None
        self.movie_matrix = None
        self.user_ids = None
        self.movie_ids = None
        self.global_mean = 3.0
        self.ratings_df = None
        
    def fit(self, ratings_df, movies_df=None):
        """Train SVD model using scikit-learn's TruncatedSVD"""
        self.ratings_df = ratings_df.copy()
        
        # Create user-movie matrix
        user_movie_matrix = ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        self.user_ids = user_movie_matrix.index.tolist()
        self.movie_ids = user_movie_matrix.columns.tolist()
        self.global_mean = ratings_df['rating'].mean()
        
        # Determine number of components
        max_components = min(len(self.user_ids), len(self.movie_ids))
        n_components = min(self.n_factors, max_components - 1)
        n_components = max(1, n_components)
        
        # Apply TruncatedSVD
        self.svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        self.user_matrix = self.svd.fit_transform(user_movie_matrix)
        self.movie_matrix = self.svd.components_.T
        
        return self
    
    def train_test_evaluate(self, ratings_df, test_size=0.2, random_state=42):
        """Evaluate the model using RMSE and MAE"""
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(ratings_df, test_size=test_size, random_state=random_state)
        self.fit(train_df)
        
        predictions = []
        actuals = []
        
        for _, row in test_df.iterrows():
            pred = self.predict_rating(row['userId'], row['movieId'])
            predictions.append(pred)
            actuals.append(row['rating'])
        
        rmse = sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        return rmse, mae, None
    
    def predict_rating(self, user_id, movie_id):
        """Predict rating using matrix factorization"""
        try:
            if user_id not in self.user_ids or movie_id not in self.movie_ids:
                return self.global_mean
            
            user_idx = self.user_ids.index(user_id)
            movie_idx = self.movie_ids.index(movie_id)
            prediction = np.dot(self.user_matrix[user_idx], self.movie_matrix[movie_idx])
            return float(np.clip(prediction, 0.5, 5.0))
        except:
            return self.global_mean
    
    def recommend_for_user(self, user_id, movies_df, ratings_df, top_n=10):
        """Get top-N recommendations for a user"""
        watched = set(ratings_df[ratings_df['userId'] == user_id]['movieId'].values)
        candidates = movies_df[~movies_df['movieId'].isin(watched)].copy()
        
        if len(candidates) == 0:
            return pd.DataFrame()
        
        candidates['predicted_rating'] = candidates['movieId'].apply(
            lambda x: self.predict_rating(user_id, x)
        )
        
        return candidates.nlargest(top_n, 'predicted_rating')[['movieId', 'title', 'genres', 'predicted_rating']].reset_index(drop=True)
    
    def save_model(self, path="models/svd_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'svd': self.svd,
                'user_matrix': self.user_matrix,
                'movie_matrix': self.movie_matrix,
                'user_ids': self.user_ids,
                'movie_ids': self.movie_ids,
                'global_mean': self.global_mean,
                'n_factors': self.n_factors
            }, f)
    
    def load_model(self, path="models/svd_model.pkl"):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.svd = data['svd']
        self.user_matrix = data['user_matrix']
        self.movie_matrix = data['movie_matrix']
        self.user_ids = data['user_ids']
        self.movie_ids = data['movie_ids']
        self.global_mean = data['global_mean']
        self.n_factors = data['n_factors']
        return self