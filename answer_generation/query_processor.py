import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np
import torch
from model_loader import load_t5_model

class QueryProcessor:
    def __init__(self, model_path="t5-base"):
        """
        Initializes the QueryProcessor with T5 model and tokenizer.

        :param model_path: Path to the pre-trained model
        """
        self.model, self.tokenizer = load_t5_model(model_path)

    def vectorize_data(self, data_frame):
        """
        Vectorizes the log data using TF-IDF.

        :param data_frame: Log data as a DataFrame
        :return: Vectors and the vectorizer object
        """
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(data_frame['LOG_CONTENT']).toarray()
        return vectors, vectorizer

    def build_faiss_index(self, vectors):
        """
        Builds a FAISS index for efficient similarity search.

        :param vectors: Log data vectors (numpy array)
        :return: FAISS index
        """
        dim = vectors.shape[1]
        faiss_index = faiss.IndexFlatL2(dim)
        faiss_index.add(np.array(vectors, dtype=np.float32))
        return faiss_index

    def find_most_accessed_page(self, data_frame):
        """
        Finds the most accessed page from the log data.

        :param data_frame: Log data as a DataFrame
        :return: The most accessed page (string)
        """
        page_counts = data_frame['URL'].value_counts()
        most_accessed_page = page_counts.idxmax()
        return most_accessed_page

    def generate_answer(self, log_data, faiss_index, query, vectorizer):
        """
        Generates an answer for a given query using the T5 model.

        :param log_data: Cleaned log data as a DataFrame
        :param faiss_index: FAISS index
        :param query: User query (string)
        :param vectorizer: TF-IDF vectorizer
        :return: Generated answer (string)
        """
        print(f"Generating answer for query: {query}")

        # Vectorize the query
        query_vector = vectorizer.transform([query]).toarray()

        # Search for the most relevant logs
        D, I = faiss_index.search(query_vector, 5)  # Retrieve top 5 results
        relevant_logs = log_data.iloc[I[0]]
        print(f"Found relevant logs: {relevant_logs}")

        # Determine the most accessed page
        most_accessed_page = self.find_most_accessed_page(log_data)

        # Generate the answer using T5
        input_text = f"The most accessed page is {most_accessed_page}. " \
                     f"Here are the relevant logs: {', '.join(relevant_logs['LOG_CONTENT'].tolist())}"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_new_tokens=50)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated answer: {answer}")

        return answer

if __name__ == "__main__":
    # Load data and models
    cleaned_data = pd.read_csv("../data/cleaned_data.csv")
    processor = QueryProcessor()
    vectors, vectorizer = processor.vectorize_data(cleaned_data)
    faiss_index = processor.build_faiss_index(vectors)

    # Test query
    test_query = "Find most accessed page"
    answer = processor.generate_answer(cleaned_data, faiss_index, test_query, vectorizer)
    print("Final answer:", answer)
