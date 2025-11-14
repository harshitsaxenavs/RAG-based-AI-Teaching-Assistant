# 🧠 RAG-Based AI Teaching Assistant

A fully local, privacy-friendly Retrieval-Augmented Generation (RAG)
system that converts your own video lectures into searchable vector
embeddings and creates a personalized AI teaching assistant. The
assistant answers questions **using only your course content**,
referencing exact videos and timestamps.

This project uses **Ollama**, **bge-m3 embeddings**, and **local LLMs**
such as Llama 3.2 or DeepSeek to keep everything **offline, free, and
secure**.

## 🚀 Features

-   Convert your video lectures into structured JSON subtitle chunks
-   Auto-extract audio from videos
-   Generate text chunks + timestamps
-   Vector embeddings using **bge-m3**
-   Embedding caching (no repeated processing)
-   RAG retrieval using cosine similarity
-   Local LLM answering with references to video 
-   Interactive Q&A loop
-   Zero API keys, 100% offline

## 📁 Project Folder Structure

    project/
    ├── videos/
    ├── audios/
    ├── jsons/
    ├── embeddings/
    ├── preprocess_json.py
    ├── llm.py
    ├── video_to_mp3.py
    ├── mp3_to_json.py
    └── README.md

## 📘 How to Use This RAG AI Teaching Assistant

### Step 1 --- Collect Your Videos

Place all your lecture videos inside **videos/**.

### Step 2 --- Convert Videos to MP3

    python video_to_mp3.py

### Step 3 --- Convert MP3 to JSON

    python mp3_to_json.py

### Step 4 --- Generate Embeddings

    python preprocess_json.py

### Step 5 --- Run the Assistant

    python inference.py

## 🛠 Requirements

    pip install numpy pandas scikit-learn joblib requests 

Install Ollama and pull models:

    OpenAI Whisper
    ollama pull bge-m3
    ollama pull llama3.2
    

## 🎥 Reference Content

This project was developed using references from **CodeWithHarry's
"Sigma Web Development Course"** on YouTube.

## 📄 License

MIT License\
Copyright (c) 2025 Harshit Saxena
