import pandas as pd
import re

def clean_data(lines):
    """
    Cleans the raw log data and converts it into a DataFrame.

    :param lines: Raw log data as a list of lines
    :return: Cleaned log data as a DataFrame
    """
    columns = ['IP', 'DATE&TIME', 'REQUEST_METHOD', 'URL', 'STATUS_CODE', 'SIZE', 'USER_AGENT', 'RANDOM_LOG_NUMBER']
    weblogs = []

    for line in lines:
        match = re.match(r'(\S+) - - \[(.*?)\] "(.*?) (.*?) (.*?)" (\d+) (\d+) "(.*?)" "(.*?)" (\d+)', line)
        if match:
            ip = match.group(1)
            datetime = match.group(2)
            request_method = match.group(3)
            url = match.group(4)
            protocol_version = match.group(5)
            status_code = match.group(6)
            size = match.group(7)
            user_agent = match.group(9)
            random_log_number = match.group(10)

            weblogs.append([ip, datetime, request_method, url, status_code, size, user_agent, random_log_number])

    weblogs = pd.DataFrame(weblogs, columns=columns)

    # Process the date and time
    weblogs['DATE&TIME'] = pd.to_datetime(weblogs['DATE&TIME'], format='%d/%b/%Y:%H:%M:%S %z')
    weblogs['DATE'] = weblogs['DATE&TIME'].dt.date
    weblogs['TIME'] = weblogs['DATE&TIME'].dt.time
    weblogs = weblogs.drop(columns=["DATE&TIME"])

    # Create LOG_CONTENT column with a concise description
    weblogs['LOG_CONTENT'] = weblogs.apply(
        lambda row: (
            f"IP {row['IP']} accessed {row['URL']} using {row['REQUEST_METHOD']} at {row['TIME']} on {row['DATE']}. "
            f"Response: {row['STATUS_CODE']}, Size: {row['SIZE']}."
        ),
        axis=1
    )

    # Remove unnecessary columns
    weblogs = weblogs.drop(columns=["USER_AGENT", "RANDOM_LOG_NUMBER"])
    return weblogs

if __name__ == "__main__":
    from file_reader import read_log_file

    file_path = "../data/web_log_data.log"
    lines = read_log_file(file_path)
    cleaned_data = clean_data(lines)
    print(cleaned_data.head())  # Print the first 5 rows to check if it works
