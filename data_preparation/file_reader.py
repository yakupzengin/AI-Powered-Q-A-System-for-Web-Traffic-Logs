import pandas as pd

def read_log_file(file_path):
    """
    Reads a log file and returns the raw data.

    :param file_path: Path to the log file
    :return: Raw log data as a list of lines
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

if __name__ == "__main__":
    file_path = "../data/web_log_data.log"
    lines = read_log_file(file_path)
    print(lines[:5])  # Print the first 5 lines to check if it works
