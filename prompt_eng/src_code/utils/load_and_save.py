import os
import json
import pandas as pd
from itertools import product
# 주석 추가
# Path 지정
config_path = "/home/prompt_eng/main/prompt_eng/src_code/config/config.json"
with open(config_path,'r') as file:
	config = json.load(file)
root_path = config['path']['root_path']
prompt_path = config['path']['prompt_path']
result_path = config['path']['result_path']
csv_path = config['path']['csv_path']
main_path = os.path.join(root_path, prompt_path)
code_path = config['path']['code_path']
code_path = os.path.join(root_path,result_path, code_path)
prompts = [str(val) for val in config['prompts']]
pt_tasks= config['pt_tasks']
components = config['components']

def extend_path(path, subpaths):
    extended_paths = []
    for subpath in subpaths:
        new_dir = os.path.join(path, subpath)
        extended_paths.append(new_dir)

    return extended_paths

def extend_path_from_list(paths, subpaths):
    extended_paths = []
    for path in paths:
        tmp_paths = extend_path(path, subpaths)
        extended_paths.extend(tmp_paths)
    return extended_paths
    

def update_components_dir():
    tmp_path = extend_path(main_path, prompts)
    tmp_path = extend_path_from_list(tmp_path,pt_tasks)
    new_dirs = extend_path_from_list(tmp_path,components)
    for dir in new_dirs:
        os.makedirs(dir, exist_ok=True)

def save_components_path(prompt_no, pt_task, component_dict):
    base_path = os.path.join(main_path,str(prompt_no),pt_task)
    for comp, comp_files in component_dict.items():
        tmp_path = os.path.join(base_path,comp)
        end_paths = extend_path(tmp_path, comp_files)
        print(end_paths)
        component_dict[comp] = end_paths
    return component_dict

def save_components_data(component_dict):
    for comp, comp_paths in component_dict.items():
        component_dict[comp] = {}
        for path in comp_paths:
            file_nm = os.path.basename(path)
            txt = read_component(path)
            component_dict[comp][file_nm] = txt
    return component_dict

def read_component(file_path):
    try:
        with open(file_path,'r',encoding='utf-8') as f:
            txt = f.read()
    except FileNotFoundError:
        raise ValueError
    return txt


def combine_dict_values(data_dict):    

    # Flatten the dictionary structure to a list of tuples (key, sub_key, value)
    flattened = [(main_key, sub_key, value) for main_key, sub_dict in data_dict.items() for sub_key, value in sub_dict.items()]

    # Generate all unique combinations of sub_keys across different main_keys
    keys_by_main_key = {main_key: list(sub_dict.keys()) for main_key, sub_dict in data_dict.items()}
    all_combinations = list(product(*keys_by_main_key.values()))

    # Prepare the data for the DataFrame
    data = []
    for combination in all_combinations:
        row = {}
        combination_values = []
        for i, main_key in enumerate(data_dict.keys()):
            sub_key = combination[i]
            value = data_dict[main_key].get(sub_key, None)
            if value:
                row[main_key] = sub_key
                combination_values.append(value)
        row['input'] = '\n'.join(combination_values)
        if row:  # Ensure the row has data
            data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Ensure all main keys are present as columns, even if no data for them in some rows
    for main_key in data_dict.keys():
        if main_key not in df.columns:
            df[main_key] = None

    # Reorder the DataFrame columns to ensure 'combination' is last
    column_order = list(data_dict.keys()) + ['input']
    df = df[column_order]

    return df



def load_data(prompt_no, pt_task,component_dict,task_file):
    tmp_dict = save_components_path(prompt_no, pt_task, component_dict)
    result_dict = save_components_data(tmp_dict)
    df = combine_dict_values(result_dict)
    task = load_task(prompt_no, pt_task,task_file)
    
    return df, task


def load_task(prompt_no, pt_task,file_nm):
    path = os.path.join(main_path,str(prompt_no),pt_task,'task',file_nm)
    task = read_component(path)

    return task

def make_py_file(user,df):
    for idx, row in df.iterrows():
        seq = row['seq']
        code_txt = row['output']
        file_nm = f"{seq}.py"
        path = os.path.join(code_path,user,file_nm)
        with open(path, 'w') as file:
            file.write(code_txt)


def save_data(df, user,prompt_no, pt_task, model_id):
    df['model_id'] = model_id
    df['user'] = user
    df['prompt_no'] = prompt_no
    df['pt_task'] = pt_task
    path = os.path.join(root_path,result_path,csv_path,user,'result.csv')
    if os.path.exists(path):
        org_df = pd.read_csv(path)
        prev_idx = len(org_df)+1
        end_idx = prev_idx + len(df)
        seq_range = range(prev_idx,end_idx)
        df['seq'] = seq_range
        if pt_task == "code-gen":
            make_py_file(user,df)
        df = pd.concat([org_df,df],axis=0,ignore_index=True)
    else:
        end_idx = len(df)+1
        df['seq'] = range(1,end_idx)
    df.to_csv(path, index=False)
 
    return df


if __name__ == "__main__":
    update_components_dir()
