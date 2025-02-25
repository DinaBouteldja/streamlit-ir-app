import numpy as np

# Load documents from documents.txt
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

# Set embedding size (512 like in the Lab 6 example)
embedding_dim = 512
num_documents = len(documents)

# Generate random embeddings (simulate real embeddings)
document_embeddings = np.random.rand(
    num_documents, embedding_dim).astype(np.float32)

# Save embeddings as a NumPy file
np.save("embeddings.npy", document_embeddings)

print(f"Generated embeddings.npy with shape: {document_embeddings.shape}")
