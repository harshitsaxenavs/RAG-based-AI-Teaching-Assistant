import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
import joblib

# ---------------- Embedding Function ----------------
def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    return r.json()["embeddings"]

# ---------------- Load / Create Embeddings ----------------
json_folder = "jsons"
embedding_folder = "embeddings"
os.makedirs(embedding_folder, exist_ok=True)

jsons = os.listdir(json_folder)
my_dicts = []
chunk_id = 0

for json_file in jsons:
    json_path = os.path.join(json_folder, json_file)
    embedding_path = os.path.join(embedding_folder, json_file)
    
    # Load existing embeddings if present
    if os.path.exists(embedding_path):
        print(f"Embeddings already exist for {json_file}, loading...")
        with open(embedding_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        for chunk in content['chunks']:
            my_dicts.append(chunk)
        continue

    # Create embeddings if not present
    with open(json_path, "r", encoding="utf-8") as f:
        content = json.load(f)
    
    print(f"Creating embeddings for {json_file}...")
    
    texts = [c['text'] for c in content['chunks']]
    batch_size = 100  # adjust if needed
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = create_embedding(batch)
        embeddings.extend(batch_embeddings)
    
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)
    
    # Save JSON with embeddings
    with open(embedding_path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)
    
    print(f"Saved embeddings for {json_file}.")

print(f"\nTotal chunks loaded: {len(my_dicts)}")

# ---------------- Ask Question ----------------
df = pd.DataFrame.from_records(my_dicts)
#Now we will save this dataframe 
joblib.dump(df,'embeddings.joblib')

