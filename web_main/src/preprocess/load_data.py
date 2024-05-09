import os
from langchain.prompts.prompt import PromptTemplate

TEMPLATE_PATH = "/home/prompt_eng/langchain/langchain_proto/web_main/data/template"


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
