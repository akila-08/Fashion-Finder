Fashion Finder
OVERVIEW
This project enables intelligent image and text-based search for fashion items using deep learning and vector similarity. It combines a CLIP-based feature extractor, Pinecone for vector search, and MongoDB/GridFS for image storage, all wrapped in a user-friendly Flask web interface.

Users can upload an image, type a text prompt, or capture an image via webcam to find visually or semantically similar fashion items. The system filters out irrelevant (non-fashion) queries using zero-shot classification and only processes fashion-related inputs.

TECHNOLOGIES USED
Backend and Web Framework
Flask: Web application and routing

Deep Learning and Feature Extraction
OpenAI CLIP (ViT-B/32): Unified vision-language model for extracting embeddings

Torch: Backend for running CLIP model

Pillow (PIL): Image processing

Vector Database
Pinecone: Fast vector similarity search and indexing of fashion image embeddings

Database and Storage
MongoDB Atlas: Cloud document database

GridFS: Binary image storage for efficient large file handling

Utility Libraries
Base64: Decoding webcam image inputs

Logging: Monitors system behavior and query flow

NumPy: Vector normalization and operations

