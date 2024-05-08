"""
* Version: 1.0
* 파일명: save_file.py
* 설명: 모델 결과 중 생성된 코드만 추출하여 지정된 위치에 저장
* 수정일자: 2024/05/02
* 수정자: 손예선
* 수정 내용
    1. 환경에 따라 경로 설정
"""

import subprocess
import re
import os

if os.name == 'posix':
    PATH = os.path.join(
        os.environ.get('HOME'),
        'langchain_proto',
        'web_main',
    )
elif os.name == 'nt':
    PATH = os.path.join(
       'c:', 
       os.environ.get('HOMEPATH'),
       'langchain_proto', 
       'web_main',
    )
else:
    PATH = None

def save_execute_python(code,path):
    """
    모델에서 생성된 코드 중 정규직으로 추출된 코드를 지정한 위치에 저장
    Args:
        code (str): 모델에서 생성된 코드
        path (str): 저장할 파일의 경로

    Returns:
        int: 처리 결과에 대한 반환 코드, 성공=0
    """
    code = re.sub('python\n','',code)
    code = re.sub(r'```\n[\w\s]*','',code).strip()
    code = re.sub('```','',code).strip()
    code = re.sub(r'\[PYTHON\]','',code)
    code = re.sub(r'\[\/PYTHON\].*','', code, flags=re.DOTALL)
    code = re.sub(r'\[/PYTHON\][\w|\s]*','',code)
    code = re.sub(
        'data.csv',
        os.path.join(PATH, 'data', 'csv', 'data.csv'),
        code
        )
    code = re.sub(
        r"pd\.read_csv\(['\"](.+?)['\"]\)",
        "pd.read_csv(os.path.join(PATH, 'data', 'csv', 'data.csv'))",
        code
        )

    with open(path, 'w', encoding='utf-8') as file:
        file.write(code)

    process = subprocess.Popen(['python3', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process.returncode
