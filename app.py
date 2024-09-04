from flask import Flask, jsonify, request, abort, send_from_directory, url_for
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import os
import pickle
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

# Create list
saved_cities = {"current_city": None, "destination_city": None}

# Load your locations data from JSON
data_path = os.path.join(os.getcwd(), 'data', 'places.json')
data = pd.read_json(data_path)

# Load or initialize visual features from VGG16
features_file = os.path.join(os.getcwd(), 'data', 'location_features_densenet.pkl')
try:
    with open(features_file, 'rb') as f:
        combined_features_dict = pickle.load(f)
except FileNotFoundError:
    combined_features_dict = {}
    
# Define the directory where images are stored
image_directory = os.path.join(os.getcwd(), 'images')

# Extract textual data (tags + about) and vectorize it
def combine_textual_data(row):
    return f"{row['tag1']} {row['tag2']} {row['tag3']} {row['tag4']} {row['about']}"

# Combine textual features for each location
data['combined_text'] = data.apply(combine_textual_data, axis=1)

# Text vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
text_features = vectorizer.fit_transform(data['combined_text'])

# Prepare the k-NN model for text features
knn_text = NearestNeighbors(n_neighbors=15, metric='cosine').fit(text_features)

@app.route('/test', methods=['GET'])    
def test():
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
#a

@app.route('/getRecommendation', methods=['POST'])
def get_recommendation():
    destination_city = saved_cities["destination_city"]
    if destination_city:
        request_data = request.json
        liked_location_ids = request_data.get('liked_location_ids', [])
        app.logger.info(f"Received liked_location_ids: {liked_location_ids}")
        
        # Liked and city data
        liked_data = data[data['place_id'].isin(liked_location_ids)]
        city_data = data[data['city'].str.lower() == destination_city.lower()]
        
        if liked_data.empty or city_data.empty:
            return jsonify([]), 200

        # 1. Dinamik olarak kullanıcı tarafından beğenilen etiketleri topla
        liked_tags = liked_data[['tag1', 'tag2', 'tag3', 'tag4']].values.flatten()
        liked_tags = [tag for tag in liked_tags if pd.notnull(tag)]  # Boş tagleri temizle

        # 2. Şehirdeki yerlerin etiketlerini kontrol ederek uygun olanları filtrele
        def filter_by_tags(row):
            place_tags = [row[col] for col in ['tag1', 'tag2', 'tag3', 'tag4'] if pd.notnull(row[col])]
            return any(tag in liked_tags for tag in place_tags)

        filtered_city_data = city_data[city_data.apply(filter_by_tags, axis=1)]
        
        if filtered_city_data.empty:
            return jsonify({"message": "No relevant locations found based on tags."}), 200

        # 3. DenseNet Görsel Özellikleri
        liked_visual_features = np.array([combined_features_dict.get(place_id, np.zeros(2048)) for place_id in liked_data['place_id']])
        city_visual_features = np.array([combined_features_dict.get(place_id, np.zeros(2048)) for place_id in filtered_city_data['place_id']])

        if len(liked_visual_features) == 0 or len(city_visual_features) == 0:
            app.logger.error("No valid visual features found for some locations.")
            return jsonify({"message": "No valid visual features found for some locations."}), 200

        # KNN görsel modelini eğit ve görsel benzerlik önerilerini al
        knn_visual = NearestNeighbors(n_neighbors=15, metric='cosine').fit(city_visual_features)
        _, visual_indices = knn_visual.kneighbors(liked_visual_features, n_neighbors=15)

        # 4. Metinsel Özellikler (Etiket + About)
        liked_text_features = vectorizer.transform(liked_data['combined_text'])
        city_text_features = vectorizer.transform(filtered_city_data['combined_text'])

        # KNN metinsel modelini eğit ve metinsel benzerlik önerilerini al
        knn_text = NearestNeighbors(n_neighbors=15, metric='cosine').fit(city_text_features)
        _, text_indices = knn_text.kneighbors(liked_text_features, n_neighbors=15)

        # 5. Metinsel ve görsel özelliklerin birleştirilmesi
        liked_combined_features = np.hstack((liked_visual_features, liked_text_features.toarray()))
        city_combined_features = np.hstack((city_visual_features, city_text_features.toarray()))

        # Birleştirilmiş özelliklerle KNN modelini eğit ve önerileri al
        knn_combined = NearestNeighbors(n_neighbors=15, metric='cosine').fit(city_combined_features)
        _, combined_indices = knn_combined.kneighbors(liked_combined_features, n_neighbors=15)

        # Görsel ve metinsel önerileri birleştir
        combined_indices = np.unique(combined_indices.flatten())
        valid_indices = [i for i in combined_indices if i < len(filtered_city_data)]

        # Önerilen lokasyonları seç
        recommended_locations = filtered_city_data.iloc[valid_indices].to_dict(orient='records')

        # Şehirdeki en yüksek rating'e sahip yerleri alın
        top_rated_places = city_data.sort_values(by='rating', ascending=False).head(5).to_dict(orient='records')

        # Yanıtı oluşturun
        response = {
            "recommended_locations": recommended_locations,
            "top_rated_places": top_rated_places  # En yüksek rating'li 5 yer
        }
        return jsonify(response)
    else:
        return jsonify({"message": "destination_city not set"}), 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)