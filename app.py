import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load precomputed document embeddings
embeddings = np.load("embeddings.npy")

# Load documents
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()


# Function to generate query embedding (Currently using a random placeholder)
def get_query_embedding(query):
    return np.random.rand(embeddings.shape[1])  
# Replace with an actual model if needed


# Retrieve top K most similar documents
def retrieve_top_k(query_embedding, embeddings, k=5):
    similarities = cosine_similarity(query_embedding.reshape(1, -1),
                                     embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]


# Streamlit UI
st.title("🔍 Information Retrieval App using Document Embeddings")

query = st.text_input("Enter your search query:")

if st.button("Search"):
    query_embedding = get_query_embedding(query)
    results = retrieve_top_k(query_embedding, embeddings)

    st.write("### Top Relevant Documents:")
    for doc, score in results:
        st.write(f"- **{doc.strip()}** (Score: {score:.4f})")