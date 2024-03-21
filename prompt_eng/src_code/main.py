from utils.lang_pipe import LangChainTest
from utils.load_and_save import load_data, save_data
from utils.load_model import LocalModelLoader, OpenaiModelLoader

def execute_pipe(user,prompt_no, pt_task, component_dict, task_file):
    df, mdl_task = load_data(prompt_no=prompt_no,
                            pt_task=pt_task,
                            component_dict=component_dict,
                            task_file=task_file)
    
    if pt_task == "kr-eng":
        loader = OpenaiModelLoader()
        model = loader.load_model(
                                  temperature= 0.0000001)
    else:
        loader = LocalModelLoader(pt_task)   
        model = loader.load_model(
                                  max_new_tokens = "10000",
                                  do_sample=True,
                                  repetition_penalty=1.1, # 중복된 결과값 통제(>1)
                                  top_k=1) 
    
    test = LangChainTest(model=model,
                        mdl_task=mdl_task)
    result_df = test.df_invoke(df)


    save_data(df=result_df,
            user=user,
            prompt_no=prompt_no,
            pt_task=pt_task)


if __name__ == "__main__":
    execute_pipe(user="ty",
			prompt_no=2,
			pt_task="kr-eng", 
			component_dict= {
										'persona':['test1.txt'],
										'format': ['test6.txt'],
										'context':['test5.txt']
							},
			task_file="test2.txt")