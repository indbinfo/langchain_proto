import os, json
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

key_path = "/home/prompt_eng/main/prompt_eng/src_code/config/key.json"

with open(key_path,'r') as file:
	config = json.load(file)

openai_key = config['key']['OPENAI_API_KEY']

llm = OpenAI(openai_api_key=openai_key)

template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
answer = llm_chain.invoke(question)
print(answer)