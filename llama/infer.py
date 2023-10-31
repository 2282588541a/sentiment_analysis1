#%%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,PeftConfig
import pandas as pd
# 例如: finetune_model_path='FlagAlpha/Llama2-Chinese-7b-Chat-LoRA'
finetune_model_path='/data/zhangxiaoming/Llama2-Chinese-main/train/sft/llama/checkpoint-400'  
config = PeftConfig.from_pretrained(finetune_model_path)
# 例如: base_model_name_or_path='meta-llama/Llama-2-7b-chat'
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,device_map='auto',torch_dtype=torch.float16,load_in_8bit=True)
model = PeftModel.from_pretrained(model, finetune_model_path, device_map={"": 0})
model =model.eval()
data_train = pd.read_csv('/data/zhangxiaoming/personal/sentiment_analysis/dev1.csv')
data_tmp=pd.DataFrame(data_train,columns=['text'])
len1=data_tmp.shape

#%%
query=[]
ans=[]
num=0
for index,row in data_tmp.iterrows():
    str1=str(getattr(row, 'text'))
    str2="Assistant: "
    pos=str1.find(str2)
    #print(len(str2))
    query.append(str1[:pos+len(str2)])
    question=str1[:pos+len(str2)]
    ans.append(str1[pos+len(str2):-5])
    data_train.loc[index,'text']=str1[pos+len(str2):-5]
    input_ids = tokenizer([question], return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')        
    generate_input = {
        "input_ids":input_ids,
        "max_new_tokens":512,
        "do_sample":True,
        "top_k":50,
        "top_p":0.95,
        "temperature":0.3,
        "repetition_penalty":1.3,
        "eos_token_id":tokenizer.eos_token_id,
        "bos_token_id":tokenizer.bos_token_id,
        "pad_token_id":tokenizer.pad_token_id
    }
    generate_ids  = model.generate(**generate_input)
    text = tokenizer.decode(generate_ids[0])
    data_train.loc[index,'pred']=text
data_train.to_csv("pred_llama.csv", index=False)

#%%    

