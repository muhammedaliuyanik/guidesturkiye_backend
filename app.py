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
        # destination_city'nin set edilip edilmediğini kontrol etmeye gerek yok, çünkü zaten if bloğunda kontrol ettik.
        
        # İstekten gelen JSON verisini alın
        request_data = request.json
        liked_location_ids = request_data.get('liked_location_ids', [])
        app.logger.info(f"Received liked_location_ids: {liked_location_ids}")

        # liked_data ve city_data değişkenlerini başlatıyoruz #1
        liked_data = pd.DataFrame()  # Boş bir DataFrame ile başlatıyoruz
        city_data = pd.DataFrame()   # Boş bir DataFrame ile başlatıyoruz

        # liked_location_ids'yi 'places.json' dosyasındaki verilerle karşılaştırın
        liked_data = data[data['place_id'].isin(liked_location_ids)]

        # Eğer liked_data boşsa, hata mesajı döndürür #2
        if liked_data.empty:
            app.logger.error("No liked locations found.")
            return jsonify({"message": "No liked locations found"}), 200

        # Şehir adına göre filtreleme yapın
        city_data = data[data['city'].str.lower() == destination_city.lower()]

        # Eğer city_data boşsa, hata mesajı döndürür #2
        if city_data.empty:
            app.logger.error("No locations found in the destination city.")
            return jsonify({"message": "No locations found in the destination city"}), 200

        # liked_data ve city_data boşsa boş bir liste döndürür
        if liked_data.empty or city_data.empty:
            return jsonify([]), 200

        # Benzerlik hesaplamaları için verileri hazırlıyoruz #3
        liked_features = np.vstack([combined_features_dict.get(place_id) for place_id in liked_data['place_id'] if combined_features_dict.get(place_id) is not None])
        city_features = np.vstack([combined_features_dict.get(place_id) for place_id in city_data['place_id'] if combined_features_dict.get(place_id) is not None])

        # Eğer feature setleri boşsa hata mesajı döndürür #4
        if liked_features.size == 0 or city_features.size == 0:
            app.logger.error("No valid features found for similarity calculation.")
            return jsonify({"message": "No valid features found"}), 200

        # Benzerlikleri hesaplayın
        similarities = cosine_similarity(liked_features, city_features)
        similarity_scores = similarities.sum(axis=0)

        # En çok benzeyen 15 lokasyonu alın
        top_n = 15
        top_n_indices = np.argsort(similarity_scores)[-top_n:]

        # En iyi 15 öneri arasından rastgele 5 tanesini seçin
        recommended_indices = np.random.choice(top_n_indices, size=5, replace=False)

        # Önerilen lokasyonları döndür
        recommended_locations = city_data.iloc[recommended_indices]
        return jsonify(recommended_locations.to_dict(orient='records'))
    else:
        # destination_city set edilmemişse hata mesajı döndür
        return jsonify({"message": "destination_city not set"}), 400



if __name__ == "__main__":
    # Update the parameters with your certificate and key file paths
    app.run(host='0.0.0.0', port=5000, debug=True)