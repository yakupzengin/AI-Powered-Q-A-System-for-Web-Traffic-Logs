def save_cleaned_data(data_frame, file_path):
    """
    Saves the cleaned data into a CSV file.

    :param data_frame: Cleaned data as a DataFrame
    :param file_path: Path to save the CSV file
    """
    data_frame.to_csv(file_path, index=False)
    print("Cleaned data has been saved to CSV.")

if __name__ == "__main__":
    from file_reader import read_log_file
    from data_cleaner import clean_data

    file_path = "../data/web_log_data.log"
    lines = read_log_file(file_path)
    cleaned_data = clean_data(lines)
    save_cleaned_data(cleaned_data, "../data/cleaned_data.csv")
