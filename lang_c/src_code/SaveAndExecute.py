import os
import re
import subprocess
from datetime import datetime
from langchain_community.tools import ShellTool


# 파이썬코드 전처리
def code_preprocessing(python_code, xlsx_path):
    # str to python_code 
    python_code = re.sub('python\n','',python_code)
    python_code = re.sub('```','',python_code).strip()

    encoding_comment = "# -*- coding: utf-8 -*-"
    matplot_lib = "import matplotlib.pyplot as plt\n"
    matplot_font = "plt.rc('font', family='NanumMyeongjo')\n"
    xlsx_code = f"df=pd.read_excel('{xlsx_path}')\n"

    # 인코딩 주석을 맨 위에 추가
    code_with_encoding = f"{encoding_comment}\n{python_code}"

    # load the dataframe code
    final_code = re.sub(rf'({matplot_lib})', rf'\1{xlsx_code}', code_with_encoding)

    # matplotlib 한글 폰트 설정을 import matplotlib.pyplot as plt 바로 아래에 추가
    final_code = re.sub(rf'({matplot_lib})', rf'\1{matplot_font}', final_code)
    
    return final_code


def generate_filename(base_name="python_code"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename =  f"{base_name}_{timestamp}.py"

    return filename


# 파일 저장 및 검증
def save_and_verify_code(filename, py_code):
    try:
        # 파일 저장
        with open(filename, "w") as file:
            file.write(py_code)
        print(f"'{filename}' 파일에 코드를 저장했습니다.")

        # 파일 절대경로 및 존재 여부 확인
        file_path = os.path.abspath(filename)
        if os.path.exists(file_path):
            print("올바른 경로에 파일이 생성되었습니다.")
        else:
            print("해당 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    except IOError as e:
        # 파일 저장 또는 읽기 실패
        print(f"파일 작업 중 오류 발생: {e}")


# .py 파일 실행
def python_file_execute(python3_path, file_path):
    try:
        # 파일 실행
        result = subprocess.run([python3_path, file_path], text=True, capture_output=True)
    
        # 실행 결과 출력
        if result.returncode == 0:
            return "성공", result.stdout
        else:
            return "실패", result.stderr
    except Exception as e:
        return "실행 중 예외 발생", str(e)