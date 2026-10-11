import sys
import numpy as np

sys.path.insert(0, "app")

from sklearn.metrics.pairwise import cosine_similarity
from semantic_rag import load_index
from semantic_retriever import create_embedding


index = load_index()

question = (
    "WHO 7 million premature deaths air pollution"
)

query_embedding = create_embedding(question)

document_embeddings = np.array([
    item["embedding"]
    for item in index
])

scores = cosine_similarity(
    [query_embedding],
    document_embeddings
)[0]

ranked_indices = np.argsort(scores)[::-1]

print("\nTOP 15 RETRIEVED CHUNKS")
print("=" * 70)

for rank, index_position in enumerate(ranked_indices[:15], start=1):
    item = index[index_position]

    print(
        f"{rank}. "
        f"{item['filename']} | "
        f"Chunk {item['chunk_number']} | "
        f"Score: {scores[index_position]:.4f}"
    )