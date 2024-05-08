from datetime import datetime
import base64
import json
import os
import time
import logging
import asyncio
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from main import response_from_llm
import nest_asyncio

nest_asyncio.apply()
# logger = logging.getLogger("main")

CONFIG_PATH = "/home/prompt_eng/langchain/langchain_proto/web_main/config/config.json"
with open(CONFIG_PATH, encoding="utf-8") as f:
    config = json.load(f)
with open(
    "/home/prompt_eng/langchain/langchain_proto/web_main/config/logging.json",
    "r",
    encoding="utf-8",
) as f:
    lc = json.load(f)
    logging.config.dictConfig(lc)
logger = logging.getLogger("main")
file_idx = datetime.now().strftime("%m%d_%H:%M")
root_path = config["path"]["root_path"]
save_dir = os.path.join(config["path"]["root_path"], config["path"]["result_path"])
# code_file = config["path"]["code_file"].format(file_idx)
# code_path = os.path.join(save_dir,config["path"]["code_path"],code_file)
logo_file = os.path.join(root_path, "bccard_logo.png")

LOADING_TXT = (
    "AI가 분석에 맞는 코드를 생성중입니다. 분석을 하는 동안 잠시만 기다려주세요."
)
DT_DESC_TXT = """
            해당 그래프는 BC 카드의 내국인 소비 데이터를 기반으로 생성하였습니다.
            내국인 소비 데이터는 일자/시간/연령대별 전국의 가맹점 소비 내역을 포함하고 있습니다.
            자세한 사항은 아래 데이터 상품을 참고해주세요."""

SUCCESS_TXT = "모델이 성공적으로 결과를 생성하였습니다."
FAIL_TXT = "파일 실행에 실패하였습니다."


async def simulate_typing_effect(text, typing_speed=0.08):
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

        # output_text += ' '
        placeholder.markdown(
            font_style + output_text + "</span>", unsafe_allow_html=True
        )


def get_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def init_page():
    st.set_page_config(
        page_title="BC카드 금융빅데이터 플랫폼 검색엔진", page_icon=logo_file
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
    init_page()

    with st.form("question_answer"):
        user_question = st.text_area(
            "질문을 입력하세요:", key="user_question", label_visibility="visible"
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
                if (
                    "last_question" not in st.session_state
                    or st.session_state.last_question != user_question
                ):
                    st.session_state.last_question = user_question
                    # response = get_response(code_return, user_question, 'wizardcoder:34b-python')
                    with st.spinner("AI가 답변을 생성중입니다.."):
                        await simulate_typing_effect(LOADING_TXT, typing_speed=0.1)
                        response, graph_path, report_path = response_from_llm(
                            user_question
                        )
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
