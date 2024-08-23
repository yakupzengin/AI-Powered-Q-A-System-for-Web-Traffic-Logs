data_file = r'C:\Users\yakupzengin\AI-Powered-Q-A-System-for-Web-Traffic-Logs\data\cleaned_data.csv'
index_file = r'C:\Users\yakupzengin\AI-Powered-Q-A-System-for-Web-Traffic-Logs\data\cleaned_data_faiss.index'

vectors_file = r'C:\Users\yakupzengin\AI-Powered-Q-A-System-for-Web-Traffic-Logs\data/vectorized_data.npy'
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import faiss
from answer_generation.query_processor import QueryProcessor

app = Flask(__name__)

# Define the file paths
data_file = r'C:\Users\yakupzengin\AI-Powered-Q-A-System-for-Web-Traffic-Logs\data\cleaned_data.csv'
index_file = r'C:\Users\yakupzengin\AI-Powered-Q-A-System-for-Web-Traffic-Logs\data\cleaned_data_faiss.index'

vectors_file = r'C:\Users\yakupzengin\AI-Powered-Q-A-System-for-Web-Traffic-Logs\data/vectorized_data.npy'

vectorizer_file = r'C:\Users\yakupzengin\AI-Powered-Q-A-System-for-Web-Traffic-Logs\data/vectorizer.pkl'  # Path to save the vectorizer

try:
    # Load the data
    log_data = pd.read_csv(data_file)
    vectors = np.load(vectors_file)  # Load vectors
    faiss_index = faiss.read_index(index_file)
    query_processor = QueryProcessor()

    # Load the vectorizer (assuming you saved it previously)
    import joblib

    vectorizer = joblib.load(vectorizer_file)
    query_processor.set_vectorizer(vectorizer)  # Set the trained vectorizer if needed

except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)


@app.route('/', methods=['GET', 'POST'])
def index():
    answer = ""
    if request.method == 'POST':
        user_query = request.form['query']
        retrieved_docs = query_processor.retrieve_documents_lists(user_query, faiss_index, vectors, log_data)
        context = " ".join(retrieved_docs['LOG_CONTENT'].tolist())
        answer = query_processor.generate_answer(user_query, context)
    return render_template('index.html', answer=answer)


if __name__ == '__main__':
    app.run(debug=True)
