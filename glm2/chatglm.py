

# %%
import copy
from transformers import  AutoModel,AutoTokenizer
import os
import wandb
wandb.login()
#以下是wandb相关参数认证
from argparse import Namespace
config1 = Namespace(
    batch_size = 4,
    lr = 1e-3,
    dropout_p = 0.1,
    )
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
model_name = "/datas/huggingface/chatglm2-6b" #或者远程 “THUDM/chatglm2-6b”
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name,trust_remote_code=True).half().cuda()


# %%
prompt = """情绪文本分类任务：将一段语句进行分类，一段语句可以属于多个分类，共分为以下八个分类 Love,Joy,Anxiety,Sorrow,Expect,Hate,Surprise,Anger

下面是一些范例:

性能比 获得 多种 测试 好评 -> Love
第一 眼 就 看出 这 刚 上市 长安 福特 蒙 迪 欧 — 致胜 想 不 实 车 比 网上 照片 和 电视 里 广告 更加 诱人 -> Love,Surprise

请对下述评论进行分类。只返回分类结果，多个分类用逗号隔开，无需其它说明和解释。

xxxxxx ->
"""

# prompt = """情绪文本分类任务：将一段语句进行分类，一段语句可以属于多个分类，共分为以下八个分类 爱、快乐、焦虑、悲伤、期望、恨、惊讶、愤怒

# 下面是一些范例:

# 性能比 获得 多种 测试 好评 -> 爱
# 第一 眼 就 看出 这 刚 上市 长安 福特 蒙 迪 欧 — 致胜 想 不 实 车 比 网上 照片 和 电视 里 广告 更加 诱人 -> 爱，惊讶

# 请对下述评论进行分类。返回分类结果，多个分类用逗号隔开，无需其它说明和解释。

# xxxxxx ->
# """

def get_prompt(text):
    return prompt.replace('xxxxxx',text)


# %%
response, his = model.chat(tokenizer, get_prompt('性能比 获得 多种 测试 好评->'), history=[])
print(response)  


# %%


# %%
#增加4个范例
his.append(("５０ 年 遇 大雪 阻断 无数 游子 回家 归途 也 影响 正常 上下班 -> ","Anxiety"))
his.append(("凑巧 那天 下午 没有 开车 而 天空 飘 起 鹅毛大雪 如果 走 着 回家 说不定 就 会 变成 雪人 就 百感交集 时候 辆 熟悉 车子 停 我的 面前 不 等 车 小柯 送 你 一程 吧 -> ","Surprise,Anxiety,Joy"))

his.append(("瞪 大 眼睛 一看 咦 这 不 可可 嘛 什么 时候 买 新车 -> ","Surprise"))
his.append(("自然 不会错 过 这么 好 机会 顺手 拉开 车门 车 内 暖和 空调 让 人 倍 感 温馨 顿时 忘却 车外 寒冷 -> ","Joy,Expect,Love"))


# %% [markdown]
# 我们来测试一下

# %%
response, history = model.chat(tokenizer, "我真的好生气", history=his)
print(response) 


# %%


# %%
#封装成一个函数吧~
def predict(text):
    response, history = model.chat(tokenizer, f"{text} ->", history=his,
    temperature=0.01)
    return response 

predict('死鬼，咋弄得这么有滋味呢') #在我们精心设计的一个评论下，ChatGLM2-6b终于预测错误了~

# %% [markdown]
# ## 一，准备数据

# %% [markdown]
# 我们需要把数据整理成对话的形式，即 context 和 target 的配对，然后拼到一起作为一条样本。
# 
# ChatGLM模型本质上做的是一个文字接龙的游戏，即给定一段话的上半部分，它会去续写下半部分。
# 
# 我们这里指定上半部分为我们设计的文本分类任务的prompt，下半部分为文本分类结果。
# 
# 所以我们微调的目标就是让它预测的下半部分跟我们的设定的文本分类一致。
# 
# 

# %% [markdown]
# ### 1，数据加载

# %%
import pandas as pd 
import numpy as np  
import datasets 
import csv



# %%
genres=[]
def before():
    data_train = pd.read_csv('Train.csv',encoding='GB18030')
    data_tmp=pd.DataFrame(data_train,columns=['ID', 'Text', 'Labels'])
    for index,row in data_tmp.iterrows():
        str2=str(getattr(row, 'Labels'))
        str2=str2.strip('[]')
        temp=(str2.split(','))
        # 进行去重合并，结果为所有特征值
        for i in temp:
            i=i.strip()
            #存在影片没有分类数据，这种直接忽略掉
            if i=='(no genres listed)':
                continue
            #正常的影片分类数据加入到分类列表里面
            if i not in genres:
                genres.append(i)
        str1=str(getattr(row, 'Labels'))
        str1 = str1.strip('"')
        new_str=str1.replace(' ',"")
        new_str=new_str.replace('[',"")
        new_str=new_str.replace(']',"")
        new_str=new_str.replace("'","")
        new_str=new_str.replace("'","")
        new_str=new_str.replace('"',"")
        new_str=new_str.replace('"',"")
        data_train.loc[index,'Labels']=new_str
    print(data_train['Labels'])
    train=data_train.iloc[1000:]
    dev=data_train.iloc[:1000]
    # dev.to_csv("test.csv", index=False)
    # train.to_csv("train.csv", index=False)
    #data_train.to_csv("train.csv",index=False)
#before()    
#print(genres)  

# %%
# #将上下文整理成与推理时候一致，参照model.chat中的源码~
# #model.build_inputs??
# data_train = pd.read_csv('train_sort.csv')
# data_tmp=pd.DataFrame(data_train,columns=['ID', 'Text', 'Labels'])

# def build_inputs(query, history,pred):
#     prompt = ""
#     num=0
#     ans=''
#     # pred=''
#     # for index,row in data_tmp.iterrows(): 
#     #     str11=str(getattr(row, 'Text'))
#     #     if(str11==query):
#     #         pred=str(getattr(row, 'Labels'))
#     #         break
#     for index,row in data_tmp.iterrows():      
#         str11=str(getattr(row, 'Labels'))
#         if(str11==pred):
#             num=num+1
#             ans=str(getattr(row, 'Text'))
#             break
#     his1=copy.deepcopy(history)
#     response, history1 = model.chat(tokenizer, ans, history=his1)
#     for i, (old_query, response) in enumerate(history1):
#         prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
#     prompt += "[Round {}]\n\n问：{} -> \n\n答：".format(len(history1) + 1, query)
#     return prompt 

def build_inputs(query, history):
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
    prompt += "[Round {}]\n\n问：{} -> \n\n答：".format(len(history) + 1, query)
    return prompt 

# %%
#print(build_inputs('味道不太行',history=his))


# %%
df=pd.read_csv("train_sort.csv")
ds_dic = datasets.Dataset.from_pandas(df).train_test_split(
    test_size = 2000,shuffle=True, seed = 43)
dftrain = ds_dic['train'].to_pandas()
dftest = ds_dic['test'].to_pandas()
#dftrain.to_parquet('data/dftrain.parquet')
#dftest.to_parquet('data/dftest.parquet')


# %%

# for i, row in dftrain.iterrows():
#     dftrain['context'][i]=(build_inputs(row['Text'], his,row['Labels']))
dftrain['context'] = [build_inputs(x,history=his) for x in dftrain['Text']]
#dftrain['context'] = [build_inputs(x, history=his, pred=pred) for x, pred in zip(dftrain['Text'], dftrain['Labels'])]
dftrain['target'] = [x for x in dftrain['Labels']]
dftrain = dftrain[['context','target']]
# dftest['context'] = [build_inputs(x, history=his, pred=pred) for x, pred in zip(dftest['Text'], dftest['Labels'])]
dftest['context'] = [build_inputs(x,history=his) for x in dftest['Text']]
dftest['target'] = [x for x in dftest['Labels']]
dftest = dftest[['context','target']]



print(dftest['context']) 

# %%
print(dftest['context'][1]) 

# %%
ds_train = datasets.Dataset.from_pandas(dftrain)
ds_val = datasets.Dataset.from_pandas(dftest)



from tqdm import tqdm
import transformers

model_name = "/datas/huggingface/chatglm-6b"
max_seq_length = 512
skip_over_length = True

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True)

config = transformers.AutoConfig.from_pretrained(
    model_name, trust_remote_code=True, device_map='auto')

def preprocess(example):
    context = example["context"]
    target = example["target"]
    
    context_ids = tokenizer.encode(
            context, 
            max_length=max_seq_length,
            truncation=True)
    
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    
    input_ids = context_ids + target_ids + [config.eos_token_id]
    
    # -100标志位后面会在计算loss时会被忽略不贡献损失，我们集中优化target部分生成的loss
    labels = [-100]*len(context_ids)+ target_ids + [config.eos_token_id]
    
    return {"input_ids": input_ids,
            "labels": labels,
            "context_len": len(context_ids),
            'target_len':len(target_ids)+1}


# %%
ds_train_token = ds_train.map(preprocess).select_columns(['input_ids','labels', 'context_len','target_len'])
if skip_over_length:
    ds_train_token = ds_train_token.filter(
        lambda example: example["context_len"]<max_seq_length and example["target_len"]<max_seq_length)

# %%
ds_val_token = ds_val.map(preprocess).select_columns(['input_ids', 'labels','context_len','target_len'])
if skip_over_length:
    ds_val_token = ds_val_token.filter(
        lambda example: example["context_len"]<max_seq_length and example["target_len"]<max_seq_length)

# %%


# %%


# %% [markdown]
# ### 3, 管道构建

# %%
def data_collator(examples: list):
    len_ids = [len(example["input_ids"]) for example in examples]
    longest = max(len_ids) #之后按照batch中最长的input_ids进行padding
    
    input_ids = []
    labels_list = []
    
    for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
        ids = example["input_ids"]
        labs = example["labels"]
        
        ids = ids + [tokenizer.pad_token_id] * (longest - length)
        labs = labs + [-100] * (longest - length)
        
        input_ids.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor(labs))
          
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }



# %%
import torch 
dl_train = torch.utils.data.DataLoader(ds_train_token,num_workers=2,batch_size=4,
                                       pin_memory=True,shuffle=True,
                                       collate_fn = data_collator)
dl_val = torch.utils.data.DataLoader(ds_val_token,num_workers=2,batch_size=4,
                                    pin_memory=True,shuffle=True,
                                     collate_fn = data_collator)


# %%
for batch in dl_train:
    break 
    

# %%
batch 

# %%
dl_train.size = 300 #用约300个step做一次验证

# %%


# %% [markdown]
# ## 二，定义模型

# %%
import warnings
warnings.filterwarnings("ignore")


# %%
from transformers import AutoTokenizer, AutoModel, TrainingArguments, AutoConfig
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

model = AutoModel.from_pretrained("/datas/huggingface/chatglm-6b",
                                  load_in_8bit=False, 
                                  trust_remote_code=True)

model.supports_gradient_checkpointing = True  #节约cuda
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False,
    r=8,
    lora_alpha=32, lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)
model.is_parallelizable = True
model.model_parallel = True
model.print_trainable_parameters()
# model.cuda(2)
device = torch.device("cuda:2")
#model = nn.DataParallel(model,device_ids=[0, 1])


# %%


# %%
from peft import get_peft_model, LoraConfig, TaskType

# %%


# %% [markdown]
# ## 三，训练模型

# %% [markdown]
# 我们使用我们的梦中情炉torchkeras来实现最优雅的训练循环~
# 
# 注意这里，为了更加高效地保存和加载参数，我们覆盖了KerasModel中的load_ckpt和save_ckpt方法，
# 
# 仅仅保存和加载lora权重，这样可以避免加载和保存全部模型权重造成的存储问题。

# %%
from torchkeras import KerasModel 
from accelerate import Accelerator 

class StepRunner:
    def __init__(self, net, loss_fn, accelerator=None, stage = "train", metrics_dict = None, 
                 optimizer = None, lr_scheduler = None
                 ):
        self.net,self.loss_fn,self.metrics_dict,self.stage = net,loss_fn,metrics_dict,stage
        self.optimizer,self.lr_scheduler = optimizer,lr_scheduler
        self.accelerator = accelerator if accelerator is not None else Accelerator() 
        if self.stage=='train':
            self.net.train() 
        else:
            self.net.eval()
    
    def __call__(self, batch):
        
        #loss
        with self.accelerator.autocast():
            loss = self.net(input_ids=batch["input_ids"],labels=batch["labels"]).loss

        #backward()
        if self.optimizer is not None and self.stage=="train":
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
        all_loss = self.accelerator.gather(loss).sum()
        
        #losses (or plain metrics that can be averaged)
        step_losses = {self.stage+"_loss":all_loss.item()}
        
        #metrics (stateful metrics)
        step_metrics = {}
        
        if self.stage=="train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses,step_metrics
    
KerasModel.StepRunner = StepRunner 


#仅仅保存lora可训练参数
def save_ckpt(self, ckpt_path='checkpoint', accelerator = None):
    unwrap_net = accelerator.unwrap_model(self.net)
    unwrap_net.save_pretrained(ckpt_path)
    
def load_ckpt(self, ckpt_path='checkpoint'):
    import os
    self.net.load_state_dict(
        torch.load(os.path.join(ckpt_path,'adapter_model.bin')),strict =False)
    self.from_scratch = False
    
KerasModel.save_ckpt = save_ckpt 
KerasModel.load_ckpt = load_ckpt 


# %%
keras_model = KerasModel(model,loss_fn = None,
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-6))
ckpt_path = 'github_chatglm4_lora_chinese'


# %%
from torchkeras.kerascallbacks import WandbCallback
wandb_cb = WandbCallback(project='glm_shot',
                         config=config1,
                         name=None,
                         save_code=True,
                         save_ckpt=True)
keras_model.fit(train_data = dl_train,
                val_data = dl_val,
                epochs=50,patience=3,
                monitor='val_loss',mode='min',
                ckpt_path = ckpt_path,
                mixed_precision='fp16',
                  callbacks = [wandb_cb]
               )


