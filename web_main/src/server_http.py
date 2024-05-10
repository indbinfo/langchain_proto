"""
* Version: 1.0
* 파일명: server_http.py
* 설명: 검색엔진 UI 구현 - 사용자가 질문을 입력하면 해당 질문에 대한 응답을 생성한 후 결과를 화면에 표시하는 모듈
* 수정일자: 2024/05/02
* 수정자: 손예선
* 수정 내용
    1. 스크립트 상단에 상수 정의
    2. 환경에 따라 경로 설정
"""
import base64
import json
import os
import time
import logging
import asyncio
from datetime import datetime

import nest_asyncio
from streamlit_extras.stylable_container import stylable_container
import streamlit as st

from main import response_from_llm

nest_asyncio.apply()

# 상수 정의
LOADING_TXT = "AI가 분석에 맞는 코드를 생성중입니다. 분석을 하는 동안 잠시만 기다려주세요."
DT_DESC_TXT = """
            해당 그래프는 BC 카드의 내국인 소비 데이터를 기반으로 생성하였습니다.
            내국인 소비 데이터는 일자/시간/연령대별 전국의 가맹점 소비 내역을 포함하고 있습니다.
            자세한 사항은 아래 데이터 상품을 참고해주세요."""

SUCCESS_TXT = "모델이 성공적으로 결과를 생성하였습니다."
FAIL_TXT = "파일 실행에 실패하였습니다."

# 환경 설정 로드
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
       "langchain_proto", 
       "web_main",
    )
else:
    PATH = None

# 설정 파일 로드
with open(os.path.join(PATH, 'config', 'config.json'),'r', encoding='utf-8') as file:
    config = json.load(file)

with open(os.path.join(PATH, 'config', 'logging.json'), 'r', encoding='utf-8') as f:
    lc = json.load(f)
    logging.config.dictConfig(lc)

logger = logging.getLogger("main")
file_idx = datetime.now().strftime("%m%d_%H:%M")
root_path = config['path']['root_path']
save_dir = os.path.join(config["path"]["root_path"],config["path"]["result_path"])
logo_file = os.path.join(root_path, 'bccard_logo.png')

async def simulate_typing_effect(text, typing_speed=.08):
    """
    텍스트 타이핑 효과

    Args:
        text (str): 타이핑 효과를 줄 텍스트
        typing_speed(float, optional): 타이핑 속도, default=.08
    """
    words = [word for word in text]
    placeholder = st.empty()
    output_text = " "
    font_style = "<span style='font-family: nanum;'>"
    for word in words:
        for letter in word:
            output_text += letter
            placeholder.markdown(
                font_style + output_text + "</span>", unsafe_allow_html=True
            )
            await asyncio.sleep(typing_speed)  # 비동기적으로 대기합니다.

        placeholder.markdown(font_style + output_text + "</span>", unsafe_allow_html=True)

def get_image(image_path):
    """
    이미지 파일을 읽고 Base64로 인코딩하여 반환

    Args:
        image_path (str): 이미지 파일 경로
    
    Returns:
        str: Base64로 인코딩된 이미지 데이터
    """
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def init_page():
    """
    페이지를 초기화하고 설정
    """
    st.set_page_config(
        page_title = 'BC카드 금융빅데이터 플랫폼 검색엔진',
        page_icon = logo_file
		)

    image_base64 = get_image(logo_file)

    st.markdown(
        f"""
    <img src="data:image/png;base64,{image_base64}" alt="Local Image" style="width: 50px; height: auto;">
    <span style="margin-left: 10px; font-size: 30px; font-weight: bold;">BC카드 금융빅데이터 플랫폼</span>
    <br><br>
    """,
        unsafe_allow_html=True,
    )

async def main_ui():
    """
    메인 UI를 초기화하고 설정
    """
    init_page()

    with st.form('question_answer'):
        user_question = st.text_area("질문을 입력하세요:",
                                     key="user_question",
                                     label_visibility='visible'
                                     )
        with stylable_container(
            "red",
            css_styles="""
            button {
                background-color: #FF0000;
                color: #F5F5F5
            }
            """,
        ):
            if st.form_submit_button(label="검색하기"):
                start_time = time.time()
                if 'last_question' not in st.session_state or \
                    st.session_state.last_question != user_question:
                    st.session_state.last_question = user_question

                    with st.spinner('AI가 답변을 생성중입니다..'):
                        await simulate_typing_effect(LOADING_TXT, typing_speed=0.1)
                        response, graph_path, report_path = response_from_llm(user_question)
                        total_time = time.time() - start_time
                        logger.info("소요 시간[모델 가동 X]: %s", total_time)
                        logger.info("response 결과: %s", response)

                    await simulate_typing_effect(response)
                    if response == SUCCESS_TXT:
                        # st.image(logo_file, use_column_width=True)
                        st.image(graph_path, use_column_width=True)
                        with open(report_path, "r", encoding="utf-8") as file:
                            report = file.read()
                        await simulate_typing_effect(report)
                        total_time = time.time() - start_time
                        logger.info("소요 시간[모델 가동 O]: %s", total_time)
                        await simulate_typing_effect(DT_DESC_TXT, typing_speed=0.1)
                        st.markdown(
                            """<table class="info-table" style="margin-left: auto; margin-right: auto;">
                                <br>
                                </br>
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
                        """,
                            unsafe_allow_html=True,
                        )
                    elif response == FAIL_TXT:
                        st.write("결과 호출에 실패하였습니다. 죄송합니다")
                        # words = ["Hello,", "this", "is", "a",
                        # "simulation", "of", "ChatGPT", "typing."]
                        # simulate_typing_effect(words, typing_speed=0.1)


if __name__ == "__main__":
    asyncio.run(main_ui())
