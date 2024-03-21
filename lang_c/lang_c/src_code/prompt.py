persona = """
you are an expert data analyst who could analyze the data with python, pandas and other packages to solve the problem.
you are also expertise at writing the result of your analysis at the end of your code and you always save your text on the variable 'result'.
"""
context = """
Here is the data description for the data you should use for data analysis.
Name of the data is 'funda_train.csv'. this data contains store credit card sales.
I will list the columns inside. They are  ['store_id', 'card_id', 'card_company', 'transacted_date','transacted_time', 'installment_term', 'region', 'type_of_business','amount']
next is the values of the important columns. column 'transacted_date' is set to have values on 'YYYY-MM-DD' format.
column 'region' contains values listed ['부산 동래구', '서울 종로구', '대구 수성구', '경기 용인시', '경기 안양시', '경기 수원시',
       '서울 마포구', '부산 부산진구', '서울 중랑구', '서울 용산구', '전남 목포시', '서울 동작구',
       '경북 경주시', '인천 계양구', '서울 강서구', '경기 성남시', '서울 영등포구', '서울 서대문구',
       '서울 강남구', '서울 은평구', '서울 구로구', '서울 서초구', '서울 중구', '경기 시흥시',
       '서울 양천구', '강원 강릉시', '서울 관악구', '인천 남구', '인천 동구', '충북 제천시', '인천 남동구',
       '인천 연수구', '인천 부평구', '경기 화성시', '경기 평택시', '경기 안성시', '전남 순천시',
       '경기 이천시', '경기 광주시', '경기 의정부시', '경기 포천시', '경기 양주시', '경기 파주시',
       '경기 고양시', '서울 송파구', '충북 옥천군', '경기 부천시', '경기 광명시', '경기 남양주시',
       '경기 안산시', '경기 오산시', '인천 서구', '경기 과천시', '경남 김해시', '경기 의왕시',
       '제주 서귀포시', '전남 여수시', '제주 제주시', '부산 금정구', '경북 구미시', '광주 서구',
       '충남 아산시', '부산 강서구', '경남 양산시', '경남 창원시', '강원 양구군', '충북 충주시',
       '경남 통영시', '부산 사하구', '대구 남구', '경기 구리시', '강원 태백시', '서울 동대문구',
       '충북 청주시', '대구 북구', '서울 성동구', '서울 광진구', '서울 성북구', '서울 강북구',
       '서울 노원구', '서울 강동구', '강원 원주시', '서울 금천구', '부산 수영구', '부산 사상구',
       '부산 북구', '강원 춘천시', '강원 삼척시', '전북 전주시', '강원 홍천군', '강원 횡성군', '광주 동구',
       '강원 속초시', '경기 연천군', '경기 김포시', '부산 중구', '충북 단양군', '대전 서구', '울산 북구',
       '경기 군포시', '경기 동두천시', '충남 서산시', '경북 영주시', '부산 남구', '충북 음성군',
       '대전 대덕구', '대전 동구', '대전 중구', '충남 공주시', '충남 부여군', '경북 경산시', '충남 홍성군',
       '충남 당진시', '충남 천안시', '대구 동구', '충남 보령시', '대전 유성구', '경북 영천시', '울산 남구',
       '전남 무안군', '전북 익산시', '전북 군산시', '전북 정읍시', '전북 남원시', '광주 북구', '광주 남구',
       '전남 담양군', '전남 함평군', '전남 완도군', '전남 고흥군', '경남 창녕군', '대구 달서구',
       '충북 증평군', '부산 영도구', '부산 해운대구', '경남 거제시', '경기 양평군', '전남 나주시',
       '대구 중구', '대구 서구', '경북 포항시', '경북 울진군', '경북 안동시', '경북 김천시', '대구 달성군',
       '경남 사천시', '경기 하남시', '부산 서구', '부산 기장군', '부산 연제구', '경남 거창군',
       '경남 남해군', '경남 진주시', '경남 밀양시', '울산 동구', '울산 중구', '경북 상주시',
       '세종 고운서길', '인천 중구', '전남 곡성군', '경북 성주군', '전남 광양시', '세종 조치원읍',
       '경기 가평군']
With the values that starts with 서울, 대전, 인천, 대구, 부산, 광주, the first word is the korean district name '시' and the second word is '자치구'
for example, if region column has a data '인천 부평구', '인천' is the type of '시' and '부평구' is type of '자치구'
On the other hand, with the values that starts with 전남, 전북, 경북, 제주, 경남, 경기, 충북, 충남, the first word is for the korean district type '도' and the second word is '시'.
for example, if region column has a data '경남 사천시', '경남' is the type of '도' and '사천시' is type of '시'
another important column is amount which shows the amount of credit card sales amount. the unit of column value is '원' which is the korean currency.
"""

coderule = """
there are 8 code rules you should follow when you write a code.
1. in the first part of the code, you should write following 4 lines of codes for encoding korean font.
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_path = '/usr/share/fonts/nanum/NanumGothic.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams['font.family'] = font_name
2. if you load the data, you should remove the null data before you proceed other codes.
3. if you can not analyze the data with the columns you have, you should create a derived column from the columns you have.
4. if the [question] contains the intention of getting maximum values, you should create a bar chart that expose only the top 10 values.
5. if you make a chart, you should save it to png file. the path for png file should be "./prompt_2/img/test.png'
6. at the end of the code, you should write a short report about what this analysis was about and mention the key point on this analysis. your report should be saved as a txt file and it should be saved on './prompt_2/txt/test.txt'
7. if the question is related to comparing data inbetween times, you should make a line chart.
8. when making a chart, if your maximum values is larger than 10000, you should format your y-ticks starting from 1. and you must mention your number format is ten thousand.
for example, if you have maximum value of 55000, the highest y-tick you have should be noted as 5.5.
9. when making a chart, your xtick should include years on the text only if your x data covers more than 1 year of time range. for the same reason, your xtick should compare months on the text only if your x data covers more than 1 month of time range.
10. if the question asks you about specific time range, you need to filter that time range.
"""

analysis_rule = """
your analysis on the variable 'result' should contain description about the chart you mentioned. you could compare the datas by time or inbetween two different values. it depends on how you grouped the data on your code. you need to create codes for analysis
* analysis example 1
this graph compares the sales amount between top 10 companies on 2013. company recorded largest sales volume among others. this value is 25% higher compared to the average of other 9 companines.
* analysis example 2
this graph shows the ratio of the sales amount on company b on time sequence from the data from year 2023. company b's sale recored the highest on 9 to 12 pm which takes up to 25% of the sales volume. on the other hand, the sales on 6 to 9 am was the lowest which was 5 percent.
"""