import json
import os

# from custom_log import setup_logger
import logging
import logging.config

from langchain_core.output_parsers import StrOutputParser
from model.load_model import OllamaModelLoader
from preprocess.load_data import load_template
from preprocess.qdrant import VectorDB, get_filter
from preprocess.save_file import save_execute_python


CONFIG_PATH = "/home/prompt_eng/langchain/langchain_proto/web_main/config/config.json"

with open(
    "/home/prompt_eng/langchain/langchain_proto/web_main/config/logging.json",
    "r",
    encoding="utf-8",
) as f:
    lc = json.load(f)
    logging.config.dictConfig(lc)
logger = logging.getLogger("main")

# FILE_IDX = datetime.now().strftime("%M")
# FILE_IDX = 1

with open(CONFIG_PATH, encoding="utf-8") as f:
    config = json.load(f)
# csv_dir = os.path.join(config["path"]["src_path"])
graph_dir = os.path.join(
    config["path"]["root_path"],
    config["path"]["result_path"],
    config["path"]["graph_path"],
)
save_dir = os.path.join(config["path"]["root_path"], config["path"]["result_path"])
# code_file = config["path"]["code_file"].format(FILE_IDX)
report_dir = os.path.join(
    config["path"]["root_path"],
    config["path"]["result_path"],
    config["path"]["report_path"],
)
# report_file = config["path"]["report_file"].format(FILE_IDX)
# graph_file = config["path"]["graph_file"].format(FILE_IDX)
# code_path = os.path.join(save_dir,code_file)

# graph_path = os.path.join(graph_dir, graph_file)
# report_path = os.path.join(report_dir, report_file)

# logger_path = config["path"]["logger_path"]
# logger = setup_logger(logger_path,'main.log')


def response_from_llm(user_prompt, FILE_IDX=1):
    code_file = config["path"]["code_file"].format(FILE_IDX)
    code_path = os.path.join(save_dir, code_file)
    report_file = config["path"]["report_file"].format(FILE_IDX)
    report_path = os.path.join(report_dir, report_file)
    graph_file = config["path"]["graph_file"].format(FILE_IDX)
    graph_path = os.path.join(graph_dir, graph_file)
    # user_prompt = "독산동의 법인카드 매출을 시간대 별로 알려줘"
    # template 로드
    code_gen_input = ["context", "report_file", "graph_file", "task"]
    code_gen_temp = load_template(
        config["template_file"]["code-gen"], input_variables=code_gen_input
    )
    # 모델 로드
    code_gen_loader = OllamaModelLoader(
        model_id="wizardcoder:34b-python", pt_task="code-gen"
    )
    code_gen_mdl = code_gen_loader.load_model()

    # qdrant 커넥션
    qdrant = VectorDB()
    qd_client = qdrant.create_connection()

    # 유/무효 질문 검증
    result = qdrant.qdrant_similarity_search(
        client=qd_client, task=user_prompt, collection_name="question", filter=None, k=5
    )

    return_txt = "요청 주신 질문에 대해서 답변이 어렵습니다."
    valid_yn = True
    for data in result:
        valid_filter = data[0].metadata["filter"]
        logger.info(valid_filter)
        print(valid_filter)
        if valid_filter == "무효질문":
            print(data[0].page_content)
            valid_yn = False
            break
    if not valid_yn:
        return return_txt, graph_path, report_path

    prompt_query_doc = qdrant.qdrant_similarity_search(
        client=qd_client, task=user_prompt, collection_name="context", k=1
    )
    # logger.info(f"prompt_query:\n{prompt_query_doc}")
    prompt_no = prompt_query_doc[0][0].metadata["filter"]
    # logger.info(f"prompt_no:{prompt_no}")
    data = qdrant.qdrant_similarity_search(
        client=qd_client,
        task=user_prompt,
        collection_name="document",
        filter=get_filter("filter", prompt_no),
        k=1,
    )
    logger.info("프롬프트 번호: %s", prompt_no)
    context = data[0][0].page_content
    # 경로 받기
    partial_prompt = code_gen_temp.partial(
        graph_file=graph_file, report_file=report_file, context=context
    )
    # qd_obj = qdrant.create_Qdrant_obj(client=qd_client, collection_name= 'format')
    # retriever = qd_obj.as_retriever(filter = get_filter("filter","format"))
    # setup_and_retrieval = RunnableParallel(
    #     {"context": retriever| format_docs, "task": RunnablePassthrough()})

    # chain = setup_and_retrieval | partial_prompt | code_gen_mdl | StrOutputParser()
    chain = partial_prompt | code_gen_mdl | StrOutputParser()
    code_txt = chain.invoke({"task": user_prompt})
    logger.info("코드 결과:\n %s", code_txt)
    prompt_result = partial_prompt.format(task=user_prompt)
    logger.info("프롬프트 결과:\n %s", prompt_result)

    return_code = save_execute_python(code_txt, code_path)
    if return_code == 0:
        return_txt = "모델이 성공적으로 결과를 생성하였습니다."
    elif return_code == 1:
        return_txt = "파일 실행에 실패하였습니다."

    return return_txt, graph_path, report_path


if __name__ == "__main__":
    response_from_llm(user_prompt="독산동의 법인카드 매출을 시간대 별로 알려줘")
# 독산동의 법인카드 매출을 시간대 별로 알려줘
