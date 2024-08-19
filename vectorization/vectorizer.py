from sklearn.feature_extraction.text import TfidfVectorizer

"""
Vectorizes the log data using TF-IDF.

:param data_frame: Log data as a DataFrame
:return: Vectors and the vectorizer object
"""
def vectorize_data(data_frame):

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data_frame['LOG_CONTENT']).toarray()
    return vectors, vectorizer

if __name__ == "__main__":
    import pandas as pd

    cleaned_data = pd.read_csv("../data/cleaned_data.csv")
    vectors, vectorizer = vectorize_data(cleaned_data)
    print(vectors.shape)  # Print the shape of the vectors to check if it works
