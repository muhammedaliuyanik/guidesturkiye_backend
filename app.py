from flask import Flask, jsonify, request, abort, send_from_directory
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import os
import pickle
import numpy as np
import logging

app = Flask(__name__)

# Create list
saved_cities = {"current_city": None, "destination_city": None}

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

# Prepare the k-NN model
if combined_features_dict:
    knn = NearestNeighbors(n_neighbors=15, algorithm='auto').fit(list(combined_features_dict.values()))
else:
    knn = None  # Handle the case where features are not available

# Convert to a list of dictionaries for the API response
locations = data.to_dict(orient='records')

# Define the directory where images are stored
image_directory = os.path.join(os.getcwd(), 'images')

@app.route('/test', methods=['GET'])    
def test():
    return jsonify({"message": "Hello, World!"})

@app.route('/images/<place_id>', methods=['GET'])
def get_image(place_id):
    try:
        image_file = f"{place_id}.png"
        image_path = os.path.join(image_directory, image_file)
        
        if not os.path.isfile(image_path):
            app.logger.error(f"Image not found at path {image_path}")
            abort(404)
        
        return send_from_directory(image_directory, image_file)
    
    except Exception as e:
        app.logger.error(f"Error retrieving image: {e}")
        abort(500)

@app.route('/getLocations', methods=['GET'])
def get_locations():
    current_city = saved_cities["current_city"]
    if current_city:
        filtered_locations = data[data['city'].str.lower() == current_city.lower()]
        return jsonify(filtered_locations.to_dict(orient='records'))
    else:
        return jsonify({"message": "current_city not set"}), 400

@app.route('/getLocationsByCityName', methods=['GET'])
def get_locations_by_city():
    global saved_cities
    current_city = request.args.get('current_city', '')
    destination_city = request.args.get('destination_city', '')
    
    saved_cities["current_city"] = current_city
    saved_cities["destination_city"] = destination_city
    
    return jsonify({"message": "Cities received successfully", "current_city": current_city, "destination_city": destination_city})

@app.route('/getRecommendation', methods=['POST'])
def get_recommendation():
    destination_city = saved_cities["destination_city"]
    if destination_city:
        request_data = request.json
        liked_location_ids = request_data.get('liked_location_ids', [])
        app.logger.info(f"Received liked_location_ids: {liked_location_ids}")
        
        liked_data = data[data['place_id'].isin(liked_location_ids)]
        city_data = data[data['city'].str.lower() == destination_city.lower()]
        
        if liked_data.empty or city_data.empty:
            return jsonify([]), 200
        
        liked_features = np.array([combined_features_dict[place_id] for place_id in liked_data['place_id']])
        city_features = np.array([combined_features_dict[place_id] for place_id in city_data['place_id']])
        
        knn.fit(city_features)
        distances, indices = knn.kneighbors(liked_features, n_neighbors=15)
        
        recommended_indices = np.unique(indices.flatten())
        recommended_locations = city_data.iloc[recommended_indices].to_dict(orient='records')
        
        filtered_places = data[data['city'].str.lower() == destination_city.lower()].to_dict(orient='records')
        top_rated_places = sorted(filtered_places, key=lambda x: float(x['rating']), reverse=True)
        
        response = {
            "recommended_locations": recommended_locations,
            "top_rated_places": top_rated_places[:5]
        }
        return jsonify(response)
    else:
        return jsonify({"message": "destination_city not set"}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
