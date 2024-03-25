from langchain.prompts.prompt import PromptTemplate
import os

template_path = "/home/prompt_eng/langchain/langchain_proto/web_main/data/template"



def load_template(template_file, input_variables):
    with open(os.path.join(template_path,template_file)) as f:
        template = f.read()
    prompt = PromptTemplate(template=template, input_variables=input_variables)

    return prompt

