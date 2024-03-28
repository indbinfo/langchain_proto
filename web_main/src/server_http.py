from datetime import datetime
import base64
import json
import os
import time
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from streamlit_extras.stylable_container import stylable_container
from main import response_from_llm
from preprocess.save_file import save_execute_python
import logging
import asyncio
import nest_asyncio

logger = logging.getLogger("main")

config_path = "/home/prompt_eng/langchain/langchain_proto/web_main/config/config.json"
with open(config_path) as f:
    config = json.load(f)

file_idx = datetime.now().strftime("%m%d_%H:%M")
root_path = config['path']['root_path']
save_dir = os.path.join(config["path"]["root_path"],config["path"]["result_path"])
# code_file = config["path"]["code_file"].format(file_idx)
# code_path = os.path.join(save_dir,config["path"]["code_path"],code_file)
logo_file = os.path.join(root_path, 'bccard_logo.png')



def simulate_typing_effect(text, typing_speed=.1):
    words = [word for word in text]
    placeholder = st.empty()
    output_text = " "
    font_style = "<span style='font-family: nanum;'>"
    for word in words:
        for letter in word:
            output_text += letter
            placeholder.markdown(font_style + output_text + "</span>", unsafe_allow_html=True)
            time.sleep(typing_speed)

        # output_text += ' '
        placeholder.markdown(font_style + output_text + "</span>", unsafe_allow_html=True)
    
def get_image(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def init_page():
    st.set_page_config(
        page_title='BC카드 PoC', 
        page_icon= logo_file
		)
    
    image_base64 = get_image(logo_file)

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


def main_ui():
    init_page()
    
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
                if 'last_question' not in st.session_state or st.session_state.last_question != user_question:
                    st.session_state.last_question = user_question
                    # response = get_response(code_return, user_question, 'wizardcoder:34b-python')
                    with st.spinner('AI가 답변을 생성중입니다.. 잠시만 기다려주세요'):
                        try:
                            start_time = time.time()
                            response, graph_path, report_path = response_from_llm(user_question)
                            total_time = time.time() - start_time
                            logger.info(f"소요 시간:{total_time}")
                        except Exception as e:
                            st.write("모델이 질문에 답변하지 못하였습니다.")
                            st.write(e)
                    
                    simulate_typing_effect(response)
                    # st.image(logo_file, use_column_width=True)
                    if not response:
                        st.write("요청 주신 질문에 대해서 답변이 어렵습니다")
                    else:
                        st.image(graph_path, use_column_width=True)
                        with open(report_path, 'r') as f:
                            report = f.read()
                        simulate_typing_effect(report)
                        st.markdown("""<table class="info-table" style="margin-left: auto; margin-right: auto;">
                                <tr>
                                    <th>제공기관</th>
                                    <th>설명</th>
                                    <th>가격</th>
                                    <th>유관상품</th>
                                </tr>
                                <tr>
                                    <td>비씨카드</td>
                                    <td>내국인 카드소비데이터</td>
                                    <td>3,000,000원</td>
                                    <td><a href="https://www.bigdata-finance.kr/dataset/datasetView.do?datastId=SET0300009" style="font-weight: bold;">바로가기</a></td>
                                </tr>
                                </table><br>
                        """, unsafe_allow_html=True
                        )
                        # words = ["Hello,", "this", "is", "a", "simulation", "of", "ChatGPT", "typing."]
                        # simulate_typing_effect(words, typing_speed=0.1)

if __name__ == "__main__":
    main_ui()
