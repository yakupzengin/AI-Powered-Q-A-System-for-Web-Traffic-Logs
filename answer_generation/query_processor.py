import torch
import faiss
import numpy as np
from answer_generation.model_loader import load_t5_model


class QueryProcessor:
    def __init__(self, model_path="t5-base"):
        """
        Initializes the QueryProcessor with T5 model and tokenizer.
        """
        self.model, self.tokenizer = load_t5_model(model_path)
        self.vectorizer = None

    def set_vectorizer(self, vectorizer):
        """
        Sets the TF-IDF vectorizer after it has been trained.

        :param vectorizer: Trained TF-IDF vectorizer
        """
        self.vectorizer = vectorizer

    def retrieve_documents_lists(self, query, index, vectors, log_data, top_k=5):
        """
        Retrieves the most relevant documents for the given query.

        :param query: User query
        :param index: FAISS index
        :param vectors: Vectorized documents
        :param log_data: DataFrame containing log data
        :param top_k: Number of top documents to retrieve
        :return: Retrieved documents DataFrame
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer is not set. Please set the vectorizer before calling retrieve_documents.")

        query_vector = self.vectorizer.transform([query]).toarray().astype(np.float32)
        distances, indices = index.search(query_vector, top_k)
        retrieved_docs = log_data.iloc[indices[0]]
        return retrieved_docs

    def generate_answer(self, query, context):
        """
        Generates an answer based on the query and context.

        :param query: User query
        :param context: Combined context from retrieved documents
        :return: Generated answer
        """
        input_text = f"question: {query} context: {context} </s>"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(input_ids)
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer
