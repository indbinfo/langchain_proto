import subprocess
import re

def save_execute_python_test(code,path):
	code = re.sub('python\n','',code)
	code = re.sub('```\n[\w\s]*','',code).strip()
	code = re.sub('```','',code).strip()
	code = re.sub('\[PYTHON\]','',code)
	code = re.sub(r'\[\/PYTHON\].*','', code, flags=re.DOTALL)
	code = re.sub('\[/PYTHON\][\w|\s]*','',code)
	code = re.sub('data.csv','/home/prompt_eng/langchain/langchain_proto/web_main/data/csv/data.csv',code)
	code = re.sub(r"pd\.read_csv\(['\"](.+?)['\"]\)",'pd.read_csv("/home/prompt_eng/langchain/langchain_proto/web_main/data/csv/data.csv")',code)
	process = subprocess.Popen(['python3', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	data = process.communicate()
	return process.returncode

if __name__ == "__main__":
	code = """
	
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 데이터프레임 로드
df = pd.read_csv("/home/prompt_eng/langchain/langchain_proto/web_main/data/csv/data.csv")


# 누락데이터 삭제
df = df.dropna(subset=['MER_ADNG_NM'])


# 컬럼 이름 변경
df.columns = ['SALE_DATE', 'TIME_CD', 'INDV_CP_DV_CD', 'MER_SIDO_NM', 'MER_CCG_NM', 'MER_ADNG_NM', 'MER_ADNG_NO', 'MAIN_BUZ_DESC', 'TP_GRP_NM', 'ALS_MER_TPBUZ_NM', 'CSTMR_SIDO_NM', 'CSTMR_CCG_NM', 'CSTMR_ADNG_NM', 'CSTMR_ADNG_NO', 'SE_CTGO_CD', 'AGE_10_CD', 'LIFE_GB_CD', 'INDV_INCM_AMT', 'MER_CNT', 'SALE_AMT', 'SALE_CNT']


# 법인 카드 필터링
df = df[df['INDV_CP_DV_CD'] == '법인']


# 인천에서 결제 비율 높은 지역구 추출
filtered_df = df[df['MER_SIDO_NM'] == '인천광역시']


# 지역구별 결제 비율 계산
payment_rate = filtered_df.groupby(['MER_CCG_NM'])['SALE_AMT'].sum() / filtered_df.groupby(['MER_CCG_NM'])['SALE_CNT'].sum()


# 결제 비율이 높은 지역구 추출
highest_payment_rate_region = payment_rate.idxmax()


# 결제 비율이 높은 지역구 데이터 추출
highest_payment_rate_data = filtered_df[filtered_df['MER_CCG_NM'] == highest_payment_rate_region]


# 그래프 생성
plt.figure(figsize=(15, 5))
sns.barplot(x=highest_payment_rate_data['MER_ADNG_NM'], y=highest_payment_rate_data['SALE_AMT'])
plt.title(f'{highest_payment_rate_region}의 결제 비율')
plt.xlabel('지역')
plt.ylabel('결제액')
plt.savefig('/home/prompt_eng/langchain/langchain_proto/web_main/data/result/graph/1_graph.png')
plt.close()

# 분석 결과 저장
with open('/home/prompt_eng/langchain/langchain_proto/web_main/data/result/text/1_text.txt', 'w') as f:
    f.write(f"{highest_payment_rate_region}의 결제 비율이 {payment_rate[highest_payment_rate_region]:.2f}이며, 가장 많은 결제액을 기록한 지역은 {highest_payment_rate_data['MER_ADNG_NM'].iloc[0]}입니다.")
"""
	path = "/home/prompt_eng/langchain/langchain_proto/web_main/data/result/1_code.py"
	data, code = save_execute_python_test(code,path)
	print(data)
	print(code)