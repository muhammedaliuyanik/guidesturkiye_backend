import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import pickle

# Function to extract BERT features from text
def extract_bert_features(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy().flatten()

# Load or initialize features
features_file = os.path.join(os.getcwd(), 'data', 'location_features.pkl')
try:
    with open(features_file, 'rb') as f:
        combined_features_dict = pickle.load(f)
except FileNotFoundError:
    combined_features_dict = {}

# Load data from JSON
data = pd.read_json(os.path.join(os.getcwd(), 'data', 'places.json'))

# Combine the tag columns and the about column into a single string
data['combined_text'] = data[['tag1', 'tag2', 'tag3', 'tag4', 'about']].fillna('').agg(' '.join, axis=1)

# Load BERT model (we will skip loading VGG16 as we are using precomputed image features)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Extract features for any new locations
for idx, row in data.iterrows():
    place_id = row['place_id']
    if place_id not in combined_features_dict:
        combined_text = row['combined_text']
        bert_features = extract_bert_features(combined_text, bert_model, tokenizer)
        
        # Assume image features are already computed in 'location_features.pkl'
        image_features = np.zeros(512)  # Placeholder if image features were not computed earlier
        
        # Combine BERT and image features
        combined_features = np.hstack([bert_features, image_features])
        combined_features_dict[place_id] = combined_features

# Save updated features (if new locations were added) ----BURDA OLMAMALI!
with open(features_file, 'wb') as f:
    pickle.dump(combined_features_dict, f)

# Selected city
def getRecommendationForCity(city_name, liked_location_ids):
    city = city_name
    city_data = data[data['city'] == city]

    # User-selected locations
    liked_locations = [liked_location_ids]
    liked_data = data[data['location_name'].isin(liked_locations)]

    # Combine features for liked locations and city locations
    liked_features = np.vstack([combined_features_dict[place_id] for place_id in liked_data['place_id']])
    city_features = np.vstack([combined_features_dict[place_id] for place_id in city_data['place_id']])

    # Compute similarities between liked locations and city locations
    similarities = cosine_similarity(liked_features, city_features)
    similarity_scores = similarities.sum(axis=0)

    # Introduce randomness in selection from top N similar locations
    top_n = 15  # Define how many top similar locations to consider
    top_n_indices = np.argsort(similarity_scores)[-top_n:]

    # Randomly select 5 out of the top 15 recommendations
    recommended_index = np.random.choice(top_n_indices, size=5, replace=False)

    # Display recommended locations with essential details
    recommended_locations = city_data.iloc[recommended_index]
    print("Recommended locations in Kayseri:")
    print(recommended_locations[['location_name', 'rating', 'comment_count', 'tag1', 'tag2', 'tag3', 'tag4', 'about']])

getRecommendationForCity()