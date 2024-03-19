import os
import json
import pandas as pd
from itertools import product
# Path 지정
config_path = "/home/prompt_eng/main/prompt_eng/src_code/config/config.json"
with open(config_path,'r') as file:
	config = json.load(file)
root_path = config['path']['root_path']
prompt_path = config['path']['prompt_path']
result_path = config['path']['result_path']
csv_path = config['path']['csv_path']
main_path = os.path.join(root_path, prompt_path)
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

def save_components_path(prompt_no, mdl_task, component_dict):
    base_path = os.path.join(main_path,str(prompt_no),mdl_task)
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
    with open(file_path,'r',encoding='utf-8') as f:
        txt = f.read()
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
        row['result'] = ' '.join(combination_values)
        if row:  # Ensure the row has data
            data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Ensure all main keys are present as columns, even if no data for them in some rows
    for main_key in data_dict.keys():
        if main_key not in df.columns:
            df[main_key] = None

    # Reorder the DataFrame columns to ensure 'combination' is last
    column_order = list(data_dict.keys()) + ['result']
    df = df[column_order]

    return df


def load_data(user,prompt_no, mdl_task,component_dict):
    tmp_dict = save_components_path(prompt_no, mdl_task, component_dict)
    result_dict = save_components_data(tmp_dict)
    new_df = combine_dict_values(result_dict)
    new_df['user'] = user
    new_df['prompt_no'] = prompt_no
    new_df['mdl_task'] = mdl_task
    path = os.path.join(root_path,result_path,csv_path,user,'result.csv')
    if os.path.exists(path):
        org_df = pd.read_csv(path)
        prev_idx = len(org_df)+1
        end_idx = prev_idx + len(new_df)
        new_df['seq'] = range(prev_idx,end_idx)
        new_df = pd.concat([org_df,new_df],axis=0,ignore_index=True)
    else:
        end_idx = len(new_df)+1
        new_df['seq'] = range(1,end_idx)
    new_df.to_csv(path, index=False)
       
    return new_df




    

if __name__ == "__main__":
    # update_components_dir()
    load_data(user='dw',prompt_no=1, mdl_task='code-gen',component_dict={'context':['1.txt','2.txt'],'exampler':['1.txt','2.txt'],'persona':['1.txt'],'task':['1.txt']})
