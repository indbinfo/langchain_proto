import os
from SaveAndExecute import code_preprocessing, generate_filename, save_and_verify_code, python_file_execute

python3_path = '/usr/local/bin/python3.11'
python_code_dir = 'python_code.py'
xlsx_dir = 'data_final.xlsx'
root_dir = '/home/langc/langchain_practice'
code_out_dir = 'prompt/output/code_out.txt'
code_out_path = os.path.join(root_dir, code_out_dir)
file_path = os.path.join(root_dir, python_code_dir)
xlsx_path = os.path.join(root_dir, xlsx_dir)

# code_out.txt
file = open(code_out_path)
code_out = file.read()

# Preprocess the code
final_code = code_preprocessing(code_out, xlsx_path)

# generate filename
filename = generate_filename()

# Save and verify the processed code
save_and_verify_code(filename, final_code)

# Execute the saved Python file
status, message = python_file_execute(python3_path, file_path)

# 코드 에러시 내용과 같이 code llm 다시 넣어줌(계획) → retry
# if status == "실패":
