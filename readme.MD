# Travel Recommender System Using Machine Learning

- This project is a content-based recommendation system that allows users to receive personalized travel location suggestions based on their preferences and interests.  
---

## Features

- Content-based recommendation engine using travel data.
- User-friendly interface built with **Streamlit**.
- Scalable architecture for integration with larger datasets or additional features.
- Natural Language Processing (NLP) for text-based recommendations.
- Lightweight and easy-to-deploy system.

---

## Demo

Click the thumbnail below to watch a quick demo of the project:  
[![Watch the video](https://img.youtube.com/vi/kjeHw0WzytQ/maxresdefault.jpg)](https://youtu.be/kjeHw0WzytQ)

---

## Requirements

To run this project, you need the following Python libraries installed:

- `pandas`
- `numpy`
- `scikit-learn`
- `nltk`
- `streamlit`
- `requests`

You can install all required libraries with a single command (see below).

---

## Installation

```bash
git clone https://github.com/yourusername/guidesturkiye-backend.git
cd guidesturkiye-backend
pip install -r requirements.txt
streamlit run src/app.py
```
 ## DevOps
- CI: GitHub Actions → flake8 + pytest (smoke tests) on push.
- Container: Dockerfile for Streamlit app.

### Quickstart (Docker)
docker build -t guides-backend .
docker run -p 8501:8501 --env-file .env guides-backend
