from flask import Flask, jsonify, request, url_for
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

app = Flask(__name__)

# Load your locations data from JSON
data_path = os.path.join(os.getcwd(), 'data', 'places.json')
data = pd.read_json(data_path)

# Load or initialize features
features_file = os.path.join(os.getcwd(), 'data', 'location_features.pkl')
try:
    with open(features_file, 'rb') as f:
        combined_features_dict = pickle.load(f)
except FileNotFoundError:
    combined_features_dict = {}

# Convert to a list of dictionaries for the API response
locations = data.to_dict(orient='records')

# Define the directory where images are stored (this can be on your local file system or in a directory in your Flask app)
image_directory = "/home/ubuntu/guidesturkiye/images"

@app.route('/test', methods=['GET'])    
def test():
    """
    Test
    """
    return jsonify({"message": "Hello, World!"})

@app.route('/images/<place_id>', methods=['GET'])
def get_image(place_id):
    try:
        # Construct the image file path
        image_file = f"{place_id}.png"
        image_path = os.path.join(image_directory, image_file)
        
        # Check if the file exists
        if not os.path.isfile(image_path):
            abort(404)  # If not found, return a 404 response
        
        # Send the image file
        return send_from_directory(image_directory, image_file)
    
    except Exception as e:
        abort(500)  # Internal server error if something goes wrong

@app.route('/getLocations', methods=['GET'])
def get_locations():
    """
    Returns a list of all locations with image URLs.
    """
    for location in locations:
        # Remove the 'image' field if it exists
        if 'image' in location:
            del location['image']

        # Generate the imageUrl
        image_filename = location.get('image_id', '')
        image_path = os.path.join(image_directory, image_filename)

        # Check if the image exists locally
        if os.path.isfile(image_path):
            location['imageUrl'] = url_for('static', filename=f'images/{image_filename}', _external=True)
        else:
            location['imageUrl'] = None  # or skip adding this key altogether

    return jsonify(locations)



@app.route('/getLocationsByCityName', methods=['GET'])
def get_locations_by_city():
    """
    Returns a list of locations filtered by city name.
    """
    city_name = request.args.get('city', '')
    filtered_locations = [place for place in locations if place['city'].lower() == city_name.lower()]
    return jsonify(filtered_locations)

@app.route('/getRecommendation', methods=['POST'])
def get_recommendations():
    """
    Returns a list of recommended locations based on the user's liked locations.
    Expects a JSON payload with 'liked_location_ids' and 'city_name'.
    """
    # İstekten gelen JSON verisini alın
    request_data = request.json
    liked_location_ids = request_data.get('liked_location_ids', [])
    city_name = request_data.get('city_name', '')

    # liked_location_ids'yi 'places.json' dosyasındaki verilerle karşılaştırın
    liked_data = data[data['place_id'].isin(liked_location_ids)]

    # Şehir adına göre filtreleme yapın
    city_data = data[data['city'].str.lower() == city_name.lower()]

    if liked_data.empty or city_data.empty:
        return jsonify([]), 200

    # Benzerlik hesaplamaları
    liked_features = np.vstack([combined_features_dict[place_id] for place_id in liked_data['place_id']])
    city_features = np.vstack([combined_features_dict[place_id] for place_id in city_data['place_id']])

    similarities = cosine_similarity(liked_features, city_features)
    similarity_scores = similarities.sum(axis=0)

    # En çok benzeyen 15 lokasyonu alın
    top_n = 15
    top_n_indices = np.argsort(similarity_scores)[-top_n:]

    # En iyi 15 öneri arasından rastgele 5 tanesini seçin
    recommended_indices = np.random.choice(top_n_indices, size=5, replace=False)

    # Önerilen lokasyonlar
    recommended_locations = city_data.iloc[recommended_indices]
    return jsonify(recommended_locations[['location_name', 'rating', 'comment_count', 'tag1', 'tag2', 'tag3', 'about']].to_dict(orient='records'))


if __name__ == "__main__":
    # Update the parameters with your certificate and key file paths
    app.run(host='0.0.0.0', port=5000)