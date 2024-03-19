from utils.lang_pipe import LangChainTest
from utils.load import load_data, save_data


df, mdl_task = load_data(prompt_no=1,
          pt_task="code-gen",
          component_dict={'context':["1.txt"],
                          'exampler':['1.txt'],
                          'persona':['1.txt']
		  					},
		  file_nm="1.txt")

print(df)

test = LangChainTest(llm='openai',
              pt_task='code-gen',
              max_tokens=10000,
              mdl_task=mdl_task)

result_df = test.df_invoke(df)


save_data(df=result_df,
          user='ty',
          prompt_no=1,
          pt_task='code-gen')
