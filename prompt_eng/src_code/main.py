from utils.lang_pipe import LangChainTest
from utils.load_and_save import load_data, save_data
from utils.load_model import OllamaModelLoader

def execute_pipe(user,prompt_no, model_id, pt_task, component_dict, task_file):
    df, mdl_task = load_data(prompt_no=prompt_no,
                            pt_task=pt_task,
                            component_dict=component_dict,
                            task_file=task_file)
    if pt_task != "code-gen":
        pt_task="kr-eng"
    loader = OllamaModelLoader(model_id=model_id, pt_task=pt_task)
    model = loader.load_model()
    
    test = LangChainTest(model=model,
                        mdl_task=mdl_task)
    result_df = test.df_invoke(df)


    save_data(df=result_df,
            model_id = model_id,
            user=user,
            prompt_no=prompt_no,
            pt_task=pt_task)


if __name__ == "__main__":
    execute_pipe(user="ty",
            model_id = "llama2:13b",
			prompt_no=2,
			pt_task="kr-eng", 
			component_dict= {
										'persona':['test1.txt'],
										'format': ['test6.txt'],
										'context':['test5.txt']
							},
			task_file="test2.txt")