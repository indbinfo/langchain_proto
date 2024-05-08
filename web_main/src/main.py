"""
* Version: 1.0
* 파일명: main.py
* 설명: 사용자 프롬프트를 입력 받아 코드를 생성하고 실행한 후 결과를 반환하는 모듈
* 수정일자: 2024/05/07
* 수정자: 손예선
* 수정 내용
    1. with open 뒤 띄어쓰기 삭제
    2. 환경에 따라 경로 설정
    3. "if valid_yn == False" -> "if valid_yn is False"
"""
import json
import os
import logging
import logging.config

from langchain_core.output_parsers import StrOutputParser
from model.load_model import OllamaModelLoader
from preprocess.load_data import load_template
from preprocess.qdrant import VectorDB, get_filter
from preprocess.save_file import save_execute_python

# 운영 체제에 따라 Config 경로 설정
if os.name == 'posix':
    CONFIG_PATH = os.path.join(
        os.environ.get('HOME'),
        'langchain_proto',
        'web_main',
        'config',
    )
elif os.name == 'nt':
    CONFIG_PATH = os.path.join(
       'C:', 
       os.environ.get('HOMEPATH'),
       "langchain_proto", 
       "web_main", 
       "config",
    )
else:
    CONFIG_PATH = None

with open(os.path.join(CONFIG_PATH, 'logging.json'),'r', encoding='utf-8') as f:
    lc = json.load(f)
    logging.config.dictConfig(lc)
logger = logging.getLogger("main")


FILE_IDX = 1
with open(os.path.join(CONFIG_PATH, 'config.json'), 'r', encoding='utf-8') as f:
    config = json.load(f)

graph_dir = os.path.join(
    config["path"]["root_path"],
    config["path"]["result_path"],
    config["path"]["graph_path"]
    )

save_dir = os.path.join(
    config["path"]["root_path"],
    config["path"]["result_path"]
    )

report_dir = os.path.join(
    config["path"]["root_path"],
    config["path"]["result_path"],
    config["path"]["report_path"]
    )

def response_from_llm(user_prompt, file_idx = 1):
    """
    유저 프롬프트를 입력 받아 코드를 생성하고 실행한 후 결과를 반환

    Args:
        user_prompt (str): 사용자가 입력한 프롬프트
        file_idx (int, optional): 결과를 저장할 파일의 인덱스. default=1
    
    Returns:
        tuple: 코드 실행 결과, 그래프 파일 경로, 리포트 파일 경로로 구성된 튜플 반환
    """
    code_file = config["path"]["code_file"].format(file_idx)
    code_path = os.path.join(save_dir,code_file)
    report_file = config["path"]["report_file"].format(file_idx)
    report_path = os.path.join(report_dir, report_file)
    graph_file = config["path"]["graph_file"].format(file_idx)
    graph_path = os.path.join(graph_dir, graph_file)

    # template 로드
    code_gen_input = ["context","report_file","graph_file","task"]
    code_gen_temp = load_template(
        config['template_file']['code-gen'],
        input_variables=code_gen_input
    )

    # 모델 로드
    code_gen_loader = OllamaModelLoader(model_id = "wizardcoder:34b-python", pt_task = "code-gen")
    code_gen_mdl = code_gen_loader.load_model()

    # qdrant 커넥션
    qdrant = VectorDB()
    qd_client = qdrant.create_connection()

    # 유/무효 질문 검증
    result = qdrant.qdrant_similarity_search(
        client = qd_client,
        task = user_prompt,
        collection_name = "question",
        filters=None,
        k=5
    )

    # 기본값 설정
    return_txt = "요청 주신 질문에 대해서 답변이 어렵습니다."
    valid_yn = True

    for data in result:
        valid_filter = data[0].metadata['filter']
        logger.info(valid_filter)
        print(valid_filter)

        if valid_filter == "무효질문":
            print(data[0].page_content)
            valid_yn = False
            break

    if valid_yn is False:
        return return_txt, graph_path, report_path

    prompt_query_doc = qdrant.qdrant_similarity_search(
        client=qd_client,
		task=user_prompt,
		collection_name="context",
	    k=1
    )

    prompt_no = prompt_query_doc[0][0].metadata['filter']

    data = qdrant.qdrant_similarity_search(
        client=qd_client,
		task=user_prompt,
		collection_name="document",
		filters=get_filter("filter", prompt_no),
		k=1,
    )

    logger.info("프롬프트 번호: %s", prompt_no)
    context = data[0][0].page_content

    # 경로 받아오기
    partial_prompt = code_gen_temp.partial(
                                        graph_file=graph_file,
                                        report_file=report_file,
                                        context = context
                                        )
    chain = partial_prompt | code_gen_mdl | StrOutputParser()
    code_txt = chain.invoke({"task":user_prompt})
    logger.info("코드 결과:\n%s", code_txt)
    prompt_result = partial_prompt.format(task=user_prompt)
    logger.info("프롬프트 결과:\n%s", prompt_result)

    return_code = save_execute_python(code_txt,code_path)
    if return_code == 0:
        return_txt = "모델이 성공적으로 결과를 생성하였습니다."
    elif return_code == 1:
        return_txt = "파일 실행에 실패하였습니다."

    return return_txt, graph_path, report_path


if __name__ == "__main__":
    response_from_llm(user_prompt="독산동의 법인카드 매출을 시간대 별로 알려줘")
