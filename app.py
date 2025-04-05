import os
import io
import torch
import clip
from flask import Flask, request, render_template, send_file
from PIL import Image
from pymongo import MongoClient
import gridfs
from pinecone import Pinecone
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId
import base64
import logging
import numpy as np

# Initialize Flask
app = Flask(__name__)

# Configure logging to show INFO and above, but enable DEBUG for relevance check
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize CLIP Model
device = "cpu"  # Force CPU for deployment
model, preprocess = clip.load("ViT-B/32", device=device)

# Connect to MongoDB (use MongoDB Atlas connection string)
client = MongoClient("mongodb+srv://Akila:Akila178@cluster0.emlei89.mongodb.net/fashion_db?retryWrites=true&w=majority&appName=Cluster0")
db = client["fashion_db"]
fs = gridfs.GridFS(db)

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_4zJ7TF_AD6fdpSsj1eMg6TiJthakzEKXg6wkfw4u4j3B5T67bqBZkBd1CFY3dVxSbHnUrg")
index = pc.Index("fashion-images")

# Create embeddings for fashion and non-fashion concepts with more specific prompts
FASHION_CONCEPT = "a photo of a clothing item"
NON_FASHION_CONCEPT = "a photo of something else"
CONCEPT_INPUTS = clip.tokenize([FASHION_CONCEPT, NON_FASHION_CONCEPT]).to(device)
with torch.no_grad():
    CONCEPT_EMBEDDINGS = model.encode_text(CONCEPT_INPUTS)
    CONCEPT_EMBEDDINGS /= CONCEPT_EMBEDDINGS.norm(dim=-1, keepdim=True)

# Minimum fashion similarity threshold
#MIN_FASHION_SIMILARITY = 0.3

# Function to extract image features
def extract_features(image_data):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
        features /= features.norm(dim=-1, keepdim=True)  # Normalize
    return features.cpu().numpy().flatten()

# Function to extract text features
def extract_text_features(text):
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        features = model.encode_text(text_input)
        features /= features.norm(dim=-1, keepdim=True)  # Normalize
    return features.cpu().numpy().flatten()

# Function to check if input is relevant to fashion using zero-shot classification
def is_relevant_to_fashion(query_vector, query_type="unknown"):
    with torch.no_grad():
        query_tensor = torch.tensor(query_vector).to(device).unsqueeze(0)
        query_tensor /= query_tensor.norm(dim=-1, keepdim=True)  # Ensure normalization
        similarities = (query_tensor @ CONCEPT_EMBEDDINGS.T).cpu().numpy().flatten()
        fashion_similarity = similarities[0]  # Similarity to "a photo of clothing or fashion item"
        non_fashion_similarity = similarities[1]  # Similarity to "a photo of something else"
        logger.info(f"Query type: {query_type}, Fashion similarity: {fashion_similarity:.4f}, Non-fashion similarity: {non_fashion_similarity:.4f}, Is fashion: {fashion_similarity > non_fashion_similarity}")
        # Query is relevant if it's more similar to fashion than non-fashion
        return fashion_similarity > non_fashion_similarity

# Route for homepage
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        query_vector = None
        query_type = "unknown"

        # Search by image upload
        if "image" in request.files:
            file = request.files["image"]
            if file.filename == "":
                return "No selected file", 400
            filename = secure_filename(file.filename)
            image_data = file.read()
            query_vector = extract_features(image_data)
            query_type = "image"
            logger.debug(f"Image query vector shape: {query_vector.shape}")

        # Search by text
        elif "text_query" in request.form:
            text_query = request.form["text_query"]
            if not text_query:
                return "No text provided", 400
            query_vector = extract_text_features(text_query)
            query_type = f"text (query: {text_query})"
            logger.debug(f"Text query vector shape: {query_vector.shape}")

        # Search by camera (image data from webcam)
        elif "camera_image" in request.form:
            camera_data = request.form["camera_image"]
            if not camera_data:
                return "No camera image provided", 400
            image_data = base64.b64decode(camera_data.split(",")[1])
            query_vector = extract_features(image_data)
            query_type = "camera"
            logger.debug(f"Camera query vector shape: {query_vector.shape}")

        else:
            return "Invalid request", 400

        # Check if the input is relevant to fashion
        if not is_relevant_to_fashion(query_vector, query_type):
            logger.info(f"Input rejected: Not relevant to fashion (type: {query_type})")
            return render_template("error.html")

        # Search for similar images in Pinecone (top 10 only)
        results = index.query(vector=query_vector.tolist(), top_k=10, include_metadata=True)
        logger.info(f"Pinecone query results: {len(results['matches'])} matches found")
        
        # Get image IDs from results
        image_ids = [match["id"] for match in results["matches"]]

        if not image_ids:
            logger.info("No matching images found in Pinecone")
            return "No matching clothing images found. Please try a different query.", 400

        return render_template("results.html", image_ids=image_ids)

    return render_template("index.html")

# Route to serve images from MongoDB
@app.route("/image/<image_id>")
def get_image(image_id):
    try:
        grid_out = fs.get(ObjectId(image_id))
        return send_file(io.BytesIO(grid_out.read()), mimetype=grid_out.content_type)
    except:
        return "Image not found", 404

# Run Flask
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)