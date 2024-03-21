from execute_pipe import execute_pipe, openai_ko2en_question

question = "부산의 동래구 중 많은 매출을 올린 업종을 막대그래프로 그려줘"

ko2en_question = openai_ko2en_question(question)

print(ko2en_question)

