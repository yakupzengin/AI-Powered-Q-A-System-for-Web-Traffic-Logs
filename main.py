import pandas as pd
import time
from data_preparation.file_reader import read_log_file
from data_preparation.data_cleaner import clean_data
from data_preparation.data_saver import save_cleaned_data
from vectorization.vectorizer import vectorize_data
from vectorization.faiss_index import build_faiss_index
from answer_generation.query_processor import QueryProcessor

if __name__ == "__main__":
    log_file_path = "data/web_log_data.log"
    cleaned_data_path = "data/cleaned_data.csv"

    # Measure time: Duration for reading the log data
    start_time = time.time()
    log_data = read_log_file(log_file_path)
    read_time = time.time() - start_time
    print(f"Time to read log data: {read_time:.4f} seconds")

    # Measure time: Duration for cleaning the data
    start_time = time.time()
    cleaned_data = clean_data(log_data)
    clean_time = time.time() - start_time
    print(f"Time to clean the data: {clean_time:.4f} seconds")

    # Measure time: Duration for saving the cleaned data
    start_time = time.time()
    save_cleaned_data(cleaned_data, cleaned_data_path)
    save_time = time.time() - start_time
    print(f"Time to save cleaned data: {save_time:.4f} seconds")

    # Measure time: Duration for vectorizing the data
    start_time = time.time()
    log_data = pd.read_csv(cleaned_data_path)
    vectors, vectorizer = vectorize_data(log_data)
    vectorize_time = time.time() - start_time
    print(f"Time to vectorize the data: {vectorize_time:.4f} seconds")

    # Measure time: Duration for building the FAISS index
    start_time = time.time()
    faiss_index = build_faiss_index(vectors)
    faiss_time = time.time() - start_time
    print(f"Time to build FAISS index: {faiss_time:.4f} seconds")

    # Measure time: Duration for processing the query
    start_time = time.time()
    query_processor = QueryProcessor()
    user_query = "What is the most accessed page?"
    answer = query_processor.generate_answer(log_data, faiss_index, user_query, vectorizer)
    answer_time = time.time() - start_time
    print(f"Time to process the query: {answer_time:.4f} seconds")

    print("Final answer:", answer)
