from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import logging
import os
import gc
import time
import psutil
from flask_cors import CORS
from collections import defaultdict
import threading
from functools import wraps

def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutError("Function timed out")]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        
        return wrapper
    return decorator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

class TimeoutError(Exception):
    """Custom exception for operation timeouts"""
    pass

# Load models with optimized memory usage
try:
    logger.info("Loading hybrid recommendation system models...")
    start_time = time.time()
    
    # Load only essential columns to reduce memory usage
    books_df = pd.read_pickle("models/books_df_slim.pkl")
    
    # Convert string columns to categorical to save memory
    for col in ['bookAuthor', 'publisher']:
        if col in books_df.columns:
            books_df[col] = books_df[col].astype('category')
    
    # Create book lookup dictionary for faster access
    books_dict = books_df.set_index('ISBN').to_dict('index')
    
    # Load only necessary model components
    svd_model = joblib.load("models/svd_model_compressed.pkl")
    item_factors = joblib.load("models/item_factors_compressed.pkl")
    item_similarities = joblib.load("models/item_similarities_compressed.pkl")
    isbn_to_idx = joblib.load("models/isbn_to_idx.pkl")
    idx_to_isbn = joblib.load("models/idx_to_isbn.pkl")
    metadata = joblib.load("models/metadata.pkl")
    
    # Report memory usage
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)
    
    load_time = time.time() - start_time
    logger.info(f"Successfully loaded {len(books_df)} books and all model components in {load_time:.2f} seconds")
    logger.info(f"Current memory usage: {memory_usage:.2f} MB")
    models_loaded = True
    
    # Force garbage collection after loading models
    gc.collect()
    
except FileNotFoundError as e:
    logger.error(f"Error: Model files not found. {str(e)}")
    models_loaded = False
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    models_loaded = False

@timeout(30)  # Set a 30-second timeout for recommendation generation
def get_hybrid_recommendations(user_ratings, n_recommendations=10):
    """
    Generate hybrid recommendations combining SVD and item-based CF with dynamic weighting.
    Similar to the BookRecommender class implementation.
    
    Args:
        user_ratings: Dictionary of {ISBN: rating} provided by the user
        n_recommendations: Number of recommendations to return
        
    Returns:
        Dictionary with recommendations and metadata
    """
    try:
        start_time = time.time()
        
        # Normalize user ratings to 0-1 scale
        user_ratings_norm = {isbn: float(rating)/10.0 for isbn, rating in user_ratings.items()}
        
        # Find valid ISBNs that exist in our model
        valid_isbns = [isbn for isbn in user_ratings_norm.keys() if isbn in isbn_to_idx]
        
        if not valid_isbns:
            return {"error": "None of the rated books exist in our database"}
            
        # Analyze user profile to determine weights
        profile_type = _analyze_user_ratings(user_ratings_norm)
        
        # Determine weights based on profile
        if profile_type == "new_user":
            # For new users - more trust to SVD
            weights = {'svd': 0.8, 'item_cf': 0.2}
        elif profile_type == "strong_preferences":
            # For users with clear preferences - more weight to item-CF
            weights = {'svd': 0.3, 'item_cf': 0.7}
        else:  # balanced_user
            weights = {'svd': 0.5, 'item_cf': 0.5}
        
        # Get recommendations from both models
        svd_scores = _get_svd_recommendations(user_ratings_norm, valid_isbns)
        item_cf_scores = _get_item_based_recommendations(user_ratings_norm, valid_isbns)
        
        # Combine all scores with weights
        all_isbns = set(svd_scores.keys()) | set(item_cf_scores.keys())
        hybrid_scores = {}
        
        for isbn in all_isbns:
            # Skip books the user has already rated
            if isbn in user_ratings_norm:
                continue
                
            # Combine scores with dynamic weights
            score = (
                svd_scores.get(isbn, 0) * weights['svd'] +
                item_cf_scores.get(isbn, 0) * weights['item_cf']
            )
            
            hybrid_scores[isbn] = score
            
        # Sort and get top recommendations
        top_isbns = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        # Format recommendations with book details
        recommendations = []
        for isbn, score in top_isbns:
            book_info = books_dict.get(isbn)
            if book_info:
                recommendations.append({
                    'ISBN': isbn,
                    'title': book_info['bookTitle'],
                    'author': book_info['bookAuthor'],
                    'year': int(book_info['yearOfPublication']) if pd.notna(book_info['yearOfPublication']) else None,
                    'publisher': book_info['publisher'],
                    'score': float(score),
                    'score_breakdown': {
                        'svd': float(svd_scores.get(isbn, 0)),
                        'item_cf': float(item_cf_scores.get(isbn, 0)),
                    },
                    'weights_used': weights
                })
        
        # Log performance metrics
        total_time = time.time() - start_time
        logger.info(f"Generated {len(recommendations)} hybrid recommendations in {total_time:.2f} seconds")
        
        return {
            "recommendations": recommendations,
            "processing_time_ms": int(total_time * 1000),
            "user_profile_type": profile_type,
            "weights_used": weights
        }
        
    except Exception as e:
        logger.error(f"Error generating hybrid recommendations: {str(e)}")
        return {"error": f"Failed to generate recommendations: {str(e)}"}

def _analyze_user_ratings(user_ratings_norm):
    """
    Analyze user ratings to determine profile type.
    Similar to BookRecommender class implementation.
    
    Args:
        user_ratings_norm: Normalized user ratings dictionary
        
    Returns:
        Profile type string: "new_user", "strong_preferences", or "balanced_user"
    """
    ratings = list(user_ratings_norm.values())
    rating_std = np.std(ratings)
    
    # Determine profile type
    if len(user_ratings_norm) <= 5:
        return "new_user"
    elif rating_std < 0.3:  # Low standard deviation = strong preferences
        return "strong_preferences"
    else:
        return "balanced_user"
    
def _get_svd_recommendations(user_ratings_norm, valid_isbns):
    """Get recommendations using SVD model"""
    # Project user ratings into latent space
    user_vector = np.zeros(svd_model.n_components)
    
    for isbn in valid_isbns:
        if isbn in isbn_to_idx:
            item_idx = isbn_to_idx[isbn]
            rating = user_ratings_norm[isbn]
            user_vector += rating * item_factors[item_idx]
    
    # Normalize user vector
    norm = np.linalg.norm(user_vector)
    if norm > 0:
        user_vector = user_vector / norm
        
    # Calculate similarity scores for all books (in batches to manage memory)
    svd_scores = {}
    batch_size = 1000
    num_items = len(idx_to_isbn)
    
    for batch_start in range(0, num_items, batch_size):
        batch_end = min(batch_start + batch_size, num_items)
        
        for idx in range(batch_start, batch_end):
            isbn = idx_to_isbn.get(idx)
            
            # Skip books the user has already rated
            if isbn in user_ratings_norm:
                continue
                
            item_vec = item_factors[idx]
            item_norm = np.linalg.norm(item_vec)
            
            if item_norm > 0:
                # Cosine similarity
                similarity = np.dot(user_vector, item_vec) / item_norm
                svd_scores[isbn] = similarity
    
    return svd_scores

def _get_item_based_recommendations(user_ratings_norm, valid_isbns):
    """Get recommendations using item-based collaborative filtering"""
    item_cf_scores = defaultdict(float)
    
    # For each book the user has rated
    for isbn in valid_isbns:
        user_rating = user_ratings_norm[isbn]
        
        # If we have similarity data for this book
        if isbn in item_similarities:
            # For each similar book
            for similar_isbn, similarity in item_similarities[isbn].items():
                # Skip books the user has already rated
                if similar_isbn in user_ratings_norm:
                    continue
                
                # Weighted sum of ratings and similarities
                item_cf_scores[similar_isbn] += user_rating * similarity
    
    # Normalize scores
    if item_cf_scores:
        max_score = max(item_cf_scores.values())
        if max_score > 0:
            item_cf_scores = {isbn: score/max_score for isbn, score in item_cf_scores.items()}
    
    return item_cf_scores

@app.route('/recommend', methods=['POST'])
def recommend():
    """API endpoint to get book recommendations"""
    if not models_loaded:
        return jsonify({
            "error": "Recommendation system not properly initialized",
            "status": "Service unavailable"
        }), 503
    
    try:
        
        start_time = time.time()
        data = request.get_json()
        
        if not data or 'ratings' not in data:
            return jsonify({"error": "No ratings provided"}), 400
        
        # Get user ratings and parameters
        user_ratings = data['ratings']
        
        # Validate ratings
        if not isinstance(user_ratings, dict):
            return jsonify({"error": "Ratings must be a dictionary with ISBN as keys and ratings as values"}), 400
        
        for isbn, rating in user_ratings.items():
            if not isinstance(rating, (int, float)) or rating < 0 or rating > 10:
                return jsonify({"error": f"Invalid rating for ISBN {isbn}. Ratings must be between 0 and 10"}), 400
        
        # Check if enough ratings provided
        if len(user_ratings) < 2:
            return jsonify({"error": "Please provide at least 2 book ratings"}), 400
        
        # Get customization parameters with defaults
        num_recommendations = int(data.get('num_recommendations', 10))
        svd_weight = float(data.get('svd_weight', 1))
        item_cf_weight = float(data.get('item_cf_weight', 2))
        
        # Validate weights
        total_weight = svd_weight + item_cf_weight
        if abs(total_weight - 1.0) > 0.001:  
            # Normalize weights to sum to 1
            factor = 1.0 / total_weight
            svd_weight *= factor
            item_cf_weight *= factor

        # Generate hybrid recommendations
        try:
            result = get_hybrid_recommendations(
                user_ratings, 
                n_recommendations=num_recommendations,
            )
            
            if "error" in result:
                return jsonify(result), 400

            # Add processing metadata
            total_time = time.time() - start_time
            result["api_processing_time_ms"] = int(total_time * 1000)
            
            return jsonify(result), 200
            
        except TimeoutError:
            logger.error("Recommendation request timed out")
            return jsonify({"error": "Request timed out - please try with fewer books or contact support"}), 408
        
    except TimeoutError:
        logger.error("Recommendation request timed out")
        return jsonify({"error": "Request timed out - please try with fewer books or contact support"}), 408
    except Exception as e:
        logger.error(f"Error processing recommendation request: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500
    
@app.route('/config', methods=['GET'])
def get_config():
    """Return configuration options for the recommendation system"""
    if not models_loaded:
        return jsonify({
            "error": "Recommendation system not properly initialized"
        }), 503
        
    return jsonify({
        "num_books": len(books_df),
        "default_settings": {
            "num_recommendations": 10,
            "svd_weight": 0.6,
            "item_cf_weight": 0.4
        },
        "parameter_ranges": {
            "num_recommendations": {"min": 1, "max": 50},
            "svd_weight": {"min": 0.0, "max": 1.0},
            "item_cf_weight": {"min": 0.0, "max": 1.0}
        }
    }), 200

if __name__ == '__main__':
    # Record start time for uptime tracking
    app.start_time = time.time()
    
    # Start the server
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"Starting hybrid recommendation service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)