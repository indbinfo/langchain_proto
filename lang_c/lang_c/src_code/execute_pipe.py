from load_model import LocalModelLoader, OpenaiModelLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def execute_pipe(pt_task):
    if pt_task == "kr-eng":
        loader = OpenaiModelLoader()
        model = loader.load_model(
                                  temperature= 0.0000001)
    else:
        loader = LocalModelLoader(pt_task)   
        model = loader.load_model(
                                  max_new_tokens = "10000",
                                  do_sample=True,
                                  repetition_penalty=1.1, # 중복된 결과값 통제(>1)
                                  top_k=1)
        
    return model

def openai_ko2en_question(question):
    template_text = f"Translate the following text to English:\n\n{question}"
    
    prompt_template = PromptTemplate.from_template(template_text)
    output_parser = StrOutputParser()

    model = execute_pipe('kr-eng')

    chain = prompt_template | model | output_parser

    ko2en_question = chain.invoke({"question":question})

    return ko2en_question

# def prompt_template(requirements, constraints, context, coderule, question):
#     template_text = f"""
#     [Requirements]{requirements}
#     [Constraints]{constraints}
#     [[Text Start]]
#     [Context]{context}
#     [coderule]{coderule}

#     """