from utils.lang_pipe import LangChainTest
from utils.load_and_save import load_data, save_data


def execute_pipe(user,prompt_no, pt_task, llm, component_dict, task_file):
    df, mdl_task = load_data(prompt_no=prompt_no,
                            pt_task=pt_task,
                            component_dict=component_dict,
                            task_file=task_file)

    test = LangChainTest(llm=llm,
                pt_task=pt_task,
                max_tokens=10000,
                mdl_task=mdl_task)

    result_df = test.df_invoke(df)


    save_data(df=result_df,
            user=user,
            prompt_no=prompt_no,
            pt_task=pt_task)


if __name__ == "__main__":
    execute_pipe(user="ty",
                prompt_no=1,
                pt_task="code-gen",
                llm = "openai", 
                component_dict= {'context':["1.txt"],
                                            'exampler':['1.txt'],
                                            'persona':['1.txt']
                                },
                task_file="1.txt")