# AI-Powered Q&A System for Web Traffic Logs
- **Project Report** : [Report](https://github.com/yakupzengin/AI-Powered-Q-A-System-for-Web-Traffic-Logs/blob/main/Yakup-Zengin-Project-Report.pdf)

## Overview

The AI-Powered Q&A System for Web Traffic Logs is an advanced system designed to process, analyze, and respond to queries about web traffic data. By leveraging state-of-the-art natural language processing and vector search technologies, this system provides accurate and insightful answers based on the web traffic logs provided.

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
   - [Data Preparation](#data-preparation)
   - [Vectorization and FAISS Index](#vectorization-and-faiss-index)
   - [Query Processing and Response Generation](#query-processing-and-response-generation)
3. [Testing](#testing)
4. [Performance Evaluation](#performance-evaluation)
5. [Improvement Recommendations](#improvement-recommendations)
6. [Conclusion](#conclusion)
7. [Appendices](#appendices)

## Introduction

This project aims to build a robust system capable of handling complex queries related to web traffic logs. By integrating various components such as data readers, cleaners, vectorizers, and a FAISS index, the system ensures efficient and accurate responses to user queries.

## System Architecture

### Data Preparation

- **Data Reader**: Reads raw web log data from a specified file.
- **Data Cleaner**: Cleans and preprocesses the raw data to remove noise and irrelevant information.
- **Data Saver**: Saves the cleaned data to a CSV file for further processing.

### Vectorization and FAISS Index

- **Vectorizer**: Converts cleaned data into vectors for efficient querying.
- **FAISS Index Builder**: Builds a FAISS index to enable fast similarity searches on the vectorized data.

### Query Processing and Response Generation

- **Query Processor**: Processes user queries and generates answers by leveraging the FAISS index and vectorized data.

## Testing

The system has been rigorously tested to ensure all components work as expected. Here are the details of the test files and their purposes:

**Test**: [Test Code ](https://github.com/yakupzengin/AI-Powered-Q-A-System-for-Web-Traffic-Logs/tree/main/test)

- **test_file_reader.py**: Tests the Data Reader component to ensure it correctly reads log data from files and handles different log formats accurately.
- **test_data_cleaner.py**: Evaluates the Data Cleaner component for its effectiveness in removing noise while preserving necessary data.
- **test_data_saver.py**: Assesses the Data Saver component to ensure cleaned data is saved correctly and in the appropriate format.
- **test_vectorizer.py**: Tests the Vectorizer component to confirm accurate conversion of cleaned data into vectors.
- **test_faiss_index.py**: Validates the FAISS Index Builder component for correct construction and performance in similarity searches.
- **test_query_processor.py**: Evaluates the Query Processor for generating relevant and accurate responses based on the FAISS index and vectorized data.
- **test_model_loader.py**: Tests the Model Loader component to ensure correct loading and readiness of the model for query processing.

## Performance Evaluation

### Time Measurements

The following time measurements reflect the duration of various stages in the system:

- **Time to Read Log Data**: 0.0020 seconds
- **Time to Clean the Data**: 0.0486 seconds
- **Time to Save Cleaned Data**: 0.0165 seconds
- **Time to Vectorize the Data**: 0.0232 seconds
- **Time to Build FAISS Index**: 0.0030 seconds
- **Time to Process the Query**: 10.2468 seconds

### Accuracy and Quality

The system generates accurate and meaningful responses to test queries. However, there are opportunities for improving performance and accuracy through various enhancements.

## Improvement Recommendations

- **Training with More Data**: Enhance model accuracy by training with a larger and more diverse dataset.
- **Model Parameter Tuning**: Optimize model parameters to improve performance and accuracy.
- **Advanced Algorithms**: Utilize more advanced algorithms and techniques to enhance overall system performance.
- **Data Quality Enhancements**: Improve the quality of raw data to achieve better results in the data cleaning process.

## Conclusion

The developed AI-powered Q&A system effectively processes web traffic data and provides accurate responses to user queries. The system's performance and accuracy have been evaluated through rigorous testing and time measurements. The proposed improvements aim to further enhance the system's capabilities.

## Appendices

- **Report**: [Project Code](https://github.com/yakupzengin/AI-Powered-Q-A-System-for-Web-Traffic-Logs/blob/main/Yakup-Zengin-Project-Report.pdf)
- **Test**: [Test ](https://github.com/yakupzengin/AI-Powered-Q-A-System-for-Web-Traffic-Logs/tree/main/test)

This README.md provides a comprehensive overview and evaluation of the project, summarizing the key components and findings. For further details or to contribute to the project, please refer to the links provided above.

**Prepared by Yakup Zengin**
