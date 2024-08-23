import pandas as pd
import time
from data_preparation.file_reader import read_log_file
from data_preparation.data_cleaner import clean_data
from data_preparation.data_saver import save_cleaned_data
from vectorization.vectorizer import vectorize_data
from vectorization.faiss_index import build_faiss_index, save_faiss_index, load_faiss_index
from answer_generation.query_processor import QueryProcessor

if __name__ == "__main__":
    log_file_path = "data/web_log_data.log"
    cleaned_data_path = "data/cleaned_data.csv"
    index_file_path = "data/cleaned_data_faiss.index"

    # Read the log data
    log_data = read_log_file(log_file_path)

    # Clean the log data
    cleaned_data = clean_data(log_data)
    save_cleaned_data(cleaned_data, cleaned_data_path)

    # Vectorize the cleaned data and build FAISS index
    log_data = pd.read_csv(cleaned_data_path)
    vectors, vectorizer = vectorize_data(log_data)
    faiss_index = build_faiss_index(vectors)
    save_faiss_index(faiss_index, index_file_path)

    # Load the FAISS index
    faiss_index = load_faiss_index(index_file_path)

    # Process user query and generate answer
    query_processor = QueryProcessor()
    query_processor.set_vectorizer(vectorizer)  # Set the trained vectorizer
    user_query = "What is the response status code for the log entry with IP address 233.223.117.90 and URL /usr/admin?"
    retrieved_docs = query_processor.retrieve_documents_lists(user_query, faiss_index, vectors, log_data)
    context = " ".join(retrieved_docs['LOG_CONTENT'].tolist())
    answer = query_processor.generate_answer(user_query, context)

    print("Final answer:", answer)
