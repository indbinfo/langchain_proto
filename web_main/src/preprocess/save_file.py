import subprocess
import re

def save_execute_python(code, path):
	code = re.sub('python\n','',code)
	code = re.sub('```\n[\w\s]*','',code).strip()
	code = re.sub('```','',code).strip()
	code = re.sub('\[PYTHON\]','',code)
	code = re.sub(r'\[\/PYTHON\].*','', code, flags=re.DOTALL)
	# code = re.sub('\[/PYTHON\][\w|\s]*','',code)
	# code = re.sub('data.csv','/home/prompt_eng/langchain/langchain_proto/web_main/data/csv/data.csv',code)
	code = re.sub(r"pd\.read_csv\(['\"](.+?)['\"]\)",'pd.read_csv("/home/prompt_eng/langchain/langchain_proto/web_main/data/csv/data.csv")',code)
	with open(path, 'w') as file:
		file.write(code)
	subprocess.Popen(['python3', path])