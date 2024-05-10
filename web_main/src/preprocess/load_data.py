"""
* Version: 1.0
* 파일명: load_data.py
* 설명: 기존 템플릿 파일을 가져오는 모듈
* 수정일자: 2024/05/02
* 수정자: 손예선
* 수정 내용
    1. 환경에 따라 경로 설정
"""

import os
from langchain.prompts.prompt import PromptTemplate

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

def load_template(template_file, input_variables):
    """파일을 불러와 프롬프트 템플릿으로 로드한다.
    Args:
        template_file (str): 불러올 템플릿 파일
        input_variables (dict): 템플릿에 들어갈 입력 변수

    Returns:
        PromptTemplate: 불러온 템플릿과 입력 변수로 초기화된 PromptTemplate 클래스의 인스턴스를 반환한다.
    """
    with open(os.path.join(TEMPLATE_PATH, template_file), encoding="utf-8") as f:
        template = f.read()
    prompt = PromptTemplate(template=template, input_variables=input_variables)

    return prompt
