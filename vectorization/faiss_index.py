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


def save_faiss_index(faiss_index, file_path):
    """
    Saves the FAISS index to a file.

    :param faiss_index: FAISS index object
    :param file_path: Path to save the FAISS index file
    """
    faiss.write_index(faiss_index, file_path)
    print("FAISS index has been saved to file.")


def load_faiss_index(file_path):
    """
    Loads the FAISS index from a file.

    :param file_path: Path to the FAISS index file
    :return: Loaded FAISS index object
    """
    return faiss.read_index(file_path)


if __name__ == "__main__":
    from vectorizer import vectorize_data
    import pandas as pd

    # Load the cleaned data
    cleaned_data = pd.read_csv("../data/cleaned_data.csv")

    # Vectorize the data
    vectors, _ = vectorize_data(cleaned_data)

    # Build the FAISS index
    faiss_index = build_faiss_index(vectors)

    # Save the FAISS index to a file
    save_faiss_index(faiss_index, "../data/cleaned_data_faiss.index")

    # Load the FAISS index from the file
    loaded_faiss_index = load_faiss_index("../data/cleaned_data_faiss.index")

    # Check if the loaded index is trained and functional
    print("Is the index trained?", loaded_faiss_index.is_trained)

    # Test a search query with FAISS index
    query_vector = np.array([vectors[0]], dtype=np.float32)  # Example: using the first vector as a query
    D, I = loaded_faiss_index.search(query_vector, k=5)  # Find the 5 nearest neighbors

    print("Query Vector:", query_vector)
    print("Nearest Neighbors (Indices):", I)
    print("Distances:", D)

    # Verify that the nearest neighbors make sense
    print("Original Vectors of Neighbors:")
    for index in I[0]:
        print(vectors[index])

    # Compare with the original vector to check correctness
    print("Original Vector:", vectors[0])
