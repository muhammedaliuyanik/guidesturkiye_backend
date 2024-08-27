from flask import Flask, jsonify, request, url_for
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from flask import Flask, abort, send_from_directory
from logging.handlers import RotatingFileHandler
import logging

app = Flask(__name__)
#create list
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

# Convert to a list of dictionaries for the API response
locations = data.to_dict(orient='records')

# Define the directory where images are stored
image_directory = os.path.join(os.getcwd(), 'images')

@app.route('/test', methods=['GET'])    
def test():
    """
    Test
    """
    return jsonify({"message": "Hello, World!"})

@app.route('/images/<place_id>', methods=['GET'])
def get_image(place_id):
    try:
        # Resim dosya yolunu oluştur
        image_file = f"{place_id}.png"  # image_file burada sadece dosya adını içerir
        image_path = os.path.join(image_directory, image_file)
        
        # Dosyanın var olup olmadığını kontrol et
        if not os.path.isfile(image_path):
            app.logger.error(f"Image not found at path {image_path}")
            abort(404)  # Eğer bulunmazsa, 404 hatası döndür
        
        # Resim dosyasını gönder
        return send_from_directory(image_directory, image_file)
    
    except Exception as e:
        app.logger.error(f"Error retrieving image: {e}")
        abort(500)  # Internal server error if something goes wrong

#Kullanıcının beğendiği Lokasyonları seçtiği ekran için bilgi dönüşü.
@app.route('/getLocations', methods=['GET'])
def get_locations():
    current_city = saved_cities["current_city"]
    if current_city:
        # Burada current_city'i kullanarak kullanıcının seçeceği lokasyonları listeliyoruz.
        filtered_locations = data[data['city'].str.lower() == current_city.lower()]
        return jsonify(filtered_locations.to_dict(orient='records'))
    else:
        return jsonify({"message": "current_city not set"}), 400

# current_city ve destination_city parametrelerini almak için API
@app.route('/getLocationsByCityName', methods=['GET'])
def get_locations_by_city():
    global saved_cities
    current_city = request.args.get('current_city', '')
    destination_city = request.args.get('destination_city', '')
    
    # Alınan şehir isimlerini sakla
    saved_cities["current_city"] = current_city
    saved_cities["destination_city"] = destination_city
    
    return jsonify({"message": "Cities received successfully", "current_city": current_city, "destination_city": destination_city})

@app.route('/getRecommendation', methods=['POST'])
def get_recommendation():
    destination_city = saved_cities["destination_city"]
    if destination_city:
        # Burada destination_city'yi kullanarak önerileri filtreleyebilir veya başka işlemler yapabilirsiniz
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
        return jsonify(recommended_locations.to_dict(orient='records'))
    else:
        return jsonify({"message": "destination_city not set"}), 400


if __name__ == "__main__":
    # Update the parameters with your certificate and key file paths
    app.run(host='0.0.0.0', port=5000)