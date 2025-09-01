# Run this once with internet connection to download and cache the model
from sentence_transformers import SentenceTransformer

print("Downloading model...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print("Model downloaded and cached successfully!")
print(f"Model cache location: {model.cache_folder}")
