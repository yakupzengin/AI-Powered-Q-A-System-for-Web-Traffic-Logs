import faiss
import numpy as np

"""
Builds a FAISS index for efficient similarity search.

:param vectors: Log data vectors (numpy array)
:return: FAISS index
"""
def build_faiss_index(vectors):

    dim = vectors.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.array(vectors, dtype=np.float32))
    return faiss_index

if __name__ == "__main__":
    from vectorizer import vectorize_data
    import pandas as pd

    cleaned_data = pd.read_csv("../data/cleaned_data.csv")
    vectors, _ = vectorize_data(cleaned_data)
    faiss_index = build_faiss_index(vectors)
    print(faiss_index.is_trained)  # Print whether the index is trained to check if it works
