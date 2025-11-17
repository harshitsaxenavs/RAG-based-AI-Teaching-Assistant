import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
import joblib
from read_chunks import create_embedding,df


df=joblib.load('embeddings.joblib')

incoming_query = input("\nAsk a Question: ").strip()
if not incoming_query:
    print("No question entered. Exiting...")
    exit()

# Create embedding for the query
question_embedding = create_embedding([incoming_query])[0]

# Compute similarities
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()

# Get top 3 results
top_results = 3
max_indx = similarities.argsort()[::-1][0:top_results]

# Add similarity scores to DataFrame
new_df = df.loc[max_indx].copy()
new_df["similarity"] = similarities[max_indx]

# Display top results with similarity
print("\nTop Results:\n")
print(new_df[["title", "number", "text", "similarity"]])
