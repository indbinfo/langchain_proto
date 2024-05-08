"""
* Version: 1.0
* 파일명: custom_log.py
* 설명: 로그 설정 및 지정된 디렉터리에 로그 파일 생성
* 수정일자: 2024/05/02
* 수정자: 손예선
* 수정 내용
    1. 모듈 Docstring 추가
    2. 함수 Docstring에 Return 설명 추가
"""
import logging
import os

def setup_logger(log_directory, log_file):
    """
    Sets up a basic logger that writes log messages to a file named app.log
    in the specified directory.

    Args:
        log_directory (str): The directory where the log file will be created.

    Returns:
        logger (logging.Logger): The configured logger object.
    """
    # Create the directory if it does not exist
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Define the path to the log file
    log_file_path = os.path.join(log_directory, log_file)

    # Create a basic logger
    logger = logging.getLogger("basicLogger")
    logger.setLevel(logging.DEBUG)  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    # Create a file handler to write log messages to the specified file
    file_handler = logging.FileHandler(log_file_path)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        '%(asctime)s %(process)d - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger
