from langchain.prompts.prompt import PromptTemplate
import os

"""
240430 손예선 - 각자의 작업 환경에 맞게 상대 경로로 세팅
"""

if os.name == 'posix':
    template_path = os.path.join(os.environ.get('HOME'), "langchain_proto", "web_main", "data", "template")
elif os.name == 'nt':
    template_path = os.path.join('c:', os.environ.get('HOMEPATH'), "langchain_proto", "web_main", "data", "template")
else:
    template_path = None

def load_template(template_file, input_variables):
    with open(os.path.join(template_path,template_file)) as f:
        template = f.read()
    prompt = PromptTemplate(template=template, input_variables=input_variables)

    return prompt