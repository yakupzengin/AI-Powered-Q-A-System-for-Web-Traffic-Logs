import pandas as pd
from data_preparation.file_reader import read_log_file
from data_preparation.data_cleaner import clean_data
from data_preparation.data_saver import save_cleaned_data
from vectorization.vectorizer import vectorize_data
from vectorization.faiss_index import build_faiss_index
from answer_generation.query_processor import QueryProcessor

if __name__ == "__main__":
    log_file_path = "data/web_log_data.log"
    cleaned_data_path = "data/cleaned_data.csv"

    # Read and clean log data
    log_data = read_log_file(log_file_path)
    cleaned_data = clean_data(log_data)
    save_cleaned_data(cleaned_data, cleaned_data_path)

    # Load the cleaned data and vectorize it
    log_data = pd.read_csv(cleaned_data_path)
    vectors, vectorizer = vectorize_data(log_data)
    faiss_index = build_faiss_index(vectors)

    # Initialize the QueryProcessor with the T5 model
    query_processor = QueryProcessor()

    # Generate an answer for the user query
    user_query = "What is the most accessed page?"
    answer = query_processor.generate_answer(log_data, faiss_index, user_query, vectorizer)
    print("Final answer:", answer)
