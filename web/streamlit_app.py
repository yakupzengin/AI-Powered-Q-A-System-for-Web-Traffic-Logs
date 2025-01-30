import streamlit as st
import numpy as np
import pandas as pd
import faiss
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from answer_generation.query_processor import QueryProcessor
import joblib

# Set page config
st.set_page_config(
    page_title="Log Analysis Q&A System",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_resources():
    # Define the file paths
    data_file= "data/cleaned_data.csv"
    index_file = "data/cleaned_data_faiss.index"
    vectors_file = "data/vectorized_data.npy"
    vectorizer_file = "data/vectorizer.pkl"

    try:
        log_data = pd.read_csv(data_file)
        vectors = np.load(vectors_file)
        faiss_index = faiss.read_index(index_file)
        query_processor = QueryProcessor()
        vectorizer = joblib.load(vectorizer_file)
        query_processor.set_vectorizer(vectorizer)
        
        return log_data, vectors, faiss_index, query_processor
    except FileNotFoundError as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None, None

def main():
    st.title("Web Traffic Log Analysis Q&A System")
    st.markdown("""
    This system helps you analyze web traffic logs by answering questions about the data.
    Simply type your question below and get instant answers!
    """)

    # Load resources
    try:
        log_data, vectors, faiss_index, query_processor = load_resources()
        
        # Check if any of the resources failed to load
        if not isinstance(log_data, pd.DataFrame) or vectors is None or faiss_index is None or query_processor is None:
            st.error("Failed to load one or more required resources. Please check the file paths and try again.")
            return

        # Create the query input
        user_query = st.text_input(
            "Ask a question about the log data:",
            placeholder="e.g., What is the response status code for IP address 233.223.117.90?"
        )

        if st.button("Get Answer", type="primary"):
            if user_query:
                with st.spinner("Generating answer..."):
                    # Process the query
                    retrieved_docs = query_processor.retrieve_documents_lists(
                        user_query, faiss_index, vectors, log_data
                    )
                    context = " ".join(retrieved_docs['LOG_CONTENT'].tolist())
                    answer = query_processor.generate_answer(user_query, context)

                    # Display results
                    st.success("Answer Generated!")
                    st.write("### Answer:")
                    st.markdown(f"**{answer}**")

                    # Display relevant log entries
                    with st.expander("View Relevant Log Entries"):
                        st.dataframe(retrieved_docs)
            else:
                st.warning("Please enter a question first.")

        # Add some helpful examples
        with st.sidebar:
            st.header("Example Questions")
            st.markdown("""
            - What is the response status code for IP address 233.223.117.90?
            - How many requests were made to /usr/admin?
            - What is the most common response status code?
            - Show me all requests from IP 192.168.1.1
            """)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure all required files and models are available.")

if __name__ == '__main__':
    main()
