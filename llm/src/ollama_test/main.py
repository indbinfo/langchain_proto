from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from streamlit_extras.stylable_container import stylable_container
import streamlit as st
import base64
import json
import asyncio
import nest_asyncio

nest_asyncio.apply()

async def simulate_typing_effect(words, typing_speed=.1):
    placeholder = st.empty()
    output_text = " "
    font_style = "<span style='font-family: nanum;'>"
    for word in words:
        for letter in word:
            output_text += letter
            placeholder.markdown(font_style + output_text + "</span>", unsafe_allow_html=True)
            await asyncio.sleep(typing_speed)  # 비동기적으로 대기합니다.

        output_text += ' '
        placeholder.markdown(font_style + output_text + "</span>", unsafe_allow_html=True)

def load_config():
    with open('/home/llm/main/llm/config/config.json', 'r') as f:
        return json.load(f)
    
def get_image(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def init_page(root_dir):
    st.set_page_config(
        page_title='BC카드 PoC', 
        page_icon='/home/llm/main/llm/src/ollama_test/bccard_logo.png')
    
    image_base64 = get_image('/home/llm/main/llm/src/ollama_test/bccard_logo.png')

    st.markdown(f"""
    <img src="data:image/png;base64,{image_base64}" alt="Local Image" style="width: 50px; height: auto;"> 
    <span style="margin-left: 10px; font-size: 30px; font-weight: bold;">BC카드 PoC</span>
    <br><br>
    """, unsafe_allow_html=True)

def create_prompt(code_return, user_question):
    template = f"""
    Translate the below text in Korean.
    User question: {user_question}
    Code return: {code_return}
    """
    return PromptTemplate.from_template(template)

def get_model(model_id):
    return Ollama(model=model_id)

def get_response(code_return, user_question, model_id):
    prompt = create_prompt(code_return, user_question)
    llm = get_model(model_id)
    
    chain = prompt | llm | StrOutputParser()
    return chain.stream({'input': user_question})

async def main_ui():
    config = load_config()
    root_dir = '/home/llm/main/llm/'
    result_path = root_dir + config['path']['result_path']

    with open(result_path + 'result_20240324145108.txt', 'r', encoding='utf-8') as f:
        code_return = f.read()
    
    init_page(root_dir)   
    
    with st.form('question_answer'):
        user_question = st.text_area("질문을 입력하세요:", key="user_question", label_visibility='visible')
        with stylable_container(
            'red',
            css_styles="""
            button {
                background-color: #FF0000;
                color: #F5F5F5
            }
            """):
            if st.form_submit_button(label='검색하기'):
                if user_question:
                    with st.spinner('실행중입니다..'):
                        response = get_response(code_return, user_question, 'wizardcoder:34b-python')
                        words = ["Hello,", "this", "is", "a", "simulation", "of", "ChatGPT", "typing."]
                        await simulate_typing_effect(words, typing_speed=0.1)  # 비동기 함수 호출
                        st.image(root_dir + 'result/corp_rate.png', use_column_width=True)
                        st.write_stream(response)

                        st.markdown("""<table class="info-table" ... </table><br>
                        """, unsafe_allow_html=True
                        )
                       
if __name__ == "__main__":
    asyncio.run(main_ui())
