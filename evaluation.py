import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

class ModelEvaluator:
    """Complete evaluation with RMSE, MAE, Precision@K, Recall@K, F1-Score"""
    
    @staticmethod
    def calculate_rmse_mae(predictions, actuals):
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        rmse = sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        return rmse, mae
    
    @staticmethod
    def precision_recall_at_k(predictions, k=10, threshold=3.5):
        """Calculate Precision@K, Recall@K, and F1-Score"""
        from collections import defaultdict
        
        user_predictions = defaultdict(list)
        
        for pred in predictions:
            user_predictions[pred.uid].append((pred.est, pred.r_ui))
        
        precisions = []
        recalls = []
        
        for user, user_preds in user_predictions.items():
            user_preds.sort(key=lambda x: x[0], reverse=True)
            top_k = user_preds[:k]
            
            relevant_in_top_k = sum(1 for _, true_r in top_k if true_r >= threshold)
            total_relevant = sum(1 for _, true_r in user_preds if true_r >= threshold)
            
            precision = relevant_in_top_k / k if k > 0 else 0
            recall = relevant_in_top_k / total_relevant if total_relevant > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return avg_precision, avg_recall, f1
    
    @staticmethod
    def evaluate_collaborative(cf_model, ratings_df, test_size=0.2):
        """Evaluate collaborative filtering model"""
        rmse, mae, predictions = cf_model.train_test_evaluate(ratings_df, test_size)
        
        # Create prediction objects for precision/recall
        from sklearn.model_selection import train_test_split
        _, test_df = train_test_split(ratings_df, test_size=test_size, random_state=42)
        
        pred_objects = []
        for _, row in test_df.iterrows():
            pred_obj = type('Prediction', (), {
                'uid': row['userId'],
                'est': cf_model.predict_rating(row['userId'], row['movieId']),
                'r_ui': row['rating']
            })()
            pred_objects.append(pred_obj)
        
        precision, recall, f1 = ModelEvaluator.precision_recall_at_k(pred_objects, k=10)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'Precision@10': precision,
            'Recall@10': recall,
            'F1-Score': f1
        }
    
    @staticmethod
    def evaluate_hybrid(hybrid_model, cf_model, content_model, test_ratings, movies, k=10):
        """Evaluate hybrid model"""
        predictions = []
        actuals = []
        
        for _, row in test_ratings.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            
            cf_score = cf_model.predict_rating(user_id, movie_id)
            
            # Get content score
            movie = movies[movies['movieId'] == movie_id]
            content_score = 0.5
            if not movie.empty:
                similar = content_model.recommend_similar_movies(movie.iloc[0]['title'], top_n=1)
                if not similar.empty:
                    content_score = similar.iloc[0]['similarity_score']
            
            predicted_rating = (0.7 * cf_score) + (0.3 * content_score * 5)
            predictions.append(predicted_rating)
            actuals.append(row['rating'])
        
        rmse, mae = ModelEvaluator.calculate_rmse_mae(predictions, actuals)
        
        # Create prediction objects
        pred_objects = []
        for i, (_, row) in enumerate(test_ratings.iterrows()):
            pred_obj = type('Prediction', (), {
                'uid': row['userId'],
                'est': predictions[i],
                'r_ui': row['rating']
            })()
            pred_objects.append(pred_obj)
        
        precision, recall, f1 = ModelEvaluator.precision_recall_at_k(pred_objects, k=k)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'Precision@10': precision,
            'Recall@10': recall,
            'F1-Score': f1
        }