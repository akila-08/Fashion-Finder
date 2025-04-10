#  Fashion Finder

## OVERVIEW

**Fashion Finder** enables intelligent image and text-based search for fashion items using deep learning and vector similarity. The project integrates:

- A **CLIP-based** feature extractor for image/text understanding
- **Pinecone** for fast vector search and similarity retrieval
- **MongoDB/GridFS** for storing and serving fashion item images
- A clean **Flask** web interface for uploading images, entering text, or using a webcam

The system also performs **zero-shot classification** to filter out non-fashion-related inputs and ensure only relevant queries are processed.

---

##  Features

-  Search similar items using:
  -  Uploaded fashion images
  -  Text descriptions (e.g., "leather jacket", "floral dress")
  -  Webcam input
-  Fashion-relevance detection using CLIP zero-shot prompts
-  Fast top-k image retrieval via vector similarity (Pinecone)
-  GridFS-backed image serving from MongoDB

---

## 🛠 TECHNOLOGIES USED

###  Backend and Web Framework
- **Flask** – Web application framework (routing, views)

###  Deep Learning and Feature Extraction
- **OpenAI CLIP (ViT-B/32)** – Vision-language model for embeddings
- **Torch (PyTorch)** – Backend for running CLIP inference
- **Pillow (PIL)** – Image preprocessing and handling

###  Vector Search and Indexing
- **Pinecone** – Vector database for fast similarity search and retrieval

###  Database and Image Storage
- **MongoDB Atlas** – Cloud-hosted NoSQL database
- **GridFS** – Efficient binary file storage and retrieval from MongoDB

###  Utility Libraries
- **Base64** – Decoding camera-captured images
- **NumPy** – Vector operations and normalization
- **Logging** – Monitoring system flow and relevance checks

---
## OUTPUT
- Here are some of the snapshots of the project
  ![image](https://github.com/user-attachments/assets/5f8fa935-2df2-4056-bba2-5ba3be0f937e)
  ![image](https://github.com/user-attachments/assets/3439c22d-640d-46b7-af01-a456ac529c68)
  ![image](https://github.com/user-attachments/assets/d5bece9c-94fa-4a6c-9eef-ccb1b4d61926)
  ![image](https://github.com/user-attachments/assets/8d569ef7-cfe9-41e1-a258-495b26f22a3c)
  ![image](https://github.com/user-attachments/assets/b296bb66-10ee-4fa6-81d2-77783828f053)



