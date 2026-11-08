import os

from fastembed import TextEmbedding


model_name = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
cache_dir = os.environ.get("MODEL_CACHE", "/models")

print(f"Caching embedding model {model_name} in {cache_dir}", flush=True)
model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
next(iter(model.embed(["Model image initialization check."])))
print("Embedding model cached successfully", flush=True)
