def safe_title_lookup(movies):
    """Get safe movie titles for dropdown"""
    if movies is not None and 'title' in movies.columns:
        return movies['title'].tolist()
    return []

def get_movie_poster_url(movie_id):
    """Get movie poster URL from TMDB (placeholder)"""
    # Using a placeholder image service
    return f"https://via.placeholder.com/150x225?text=Movie+{movie_id}"