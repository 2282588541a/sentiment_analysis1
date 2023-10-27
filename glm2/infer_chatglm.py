from peft import PeftModel 
from transformers import  AutoModel,AutoTokenizer
import os
import pandas as pd
import copy
#os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # 指定GPU 0 和 GPU 1

model = AutoModel.from_pretrained("/datas/huggingface/chatglm-6b",
                                  load_in_8bit=False, 
                                  trust_remote_code=True, 
                                  device_map='auto')
model = PeftModel.from_pretrained(model,'/data/zhangxiaoming/personal/Mistral_lora/github_chatglm4_lora_chinese')
model = model.merge_and_unload() #合并lora权重
model.half().cuda()
# model_name = "/datas/huggingface/chatglm2-6b" #或者远程 “THUDM/chatglm2-6b”
tokenizer = AutoTokenizer.from_pretrained(
    '/datas/huggingface/chatglm-6b', trust_remote_code=True)
# model = AutoModel.from_pretrained(model_name,trust_remote_code=True).half().cuda(3)

prompt = """情绪文本分类任务：将一段语句进行分类，一段语句可以属于多个分类，共分为以下八个分类 Love,Joy,Anxiety,Sorrow,Expect,Hate,Surprise,Anger

下面是一些范例:

性能比 获得 多种 测试 好评 -> Love
第一 眼 就 看出 这 刚 上市 长安 福特 蒙 迪 欧 — 致胜 想 不 实 车 比 网上 照片 和 电视 里 广告 更加 诱人 -> Love,Surprise

请对下述评论进行分类。只返回分类结果，多个分类用逗号隔开，无需其它说明和解释。

xxxxxx ->
"""
train1=pd.DataFrame()
genres=[]
def before():
    data_train = pd.read_csv('Train.csv',encoding='GB18030')
    data_tmp=pd.DataFrame(data_train,columns=['ID', 'Text', 'Labels'])
    for index,row in data_tmp.iterrows():      
        str1=str(getattr(row, 'Labels'))
        str1 = str1.strip('"')
        new_str=str1.replace(' ',"")
        new_str=new_str.replace('[',"")
        new_str=new_str.replace(']',"")
        new_str=new_str.replace("'","")
        new_str=new_str.replace("'","")
        new_str=new_str.replace('"',"")
        new_str=new_str.replace('"',"")
        temp=(new_str.split(','))
        # 进行去重合并，结果为所有特征值
        for i in temp:
            i=i.strip()
            #存在影片没有分类数据，这种直接忽略掉
            if i=='(no genres listed)':
                continue
            #正常的影片分类数据加入到分类列表里面
            if i not in genres:
                genres.append(i) 

        new_ans=''

        data_train.loc[index,'Labels']=new_str
    for index,row in data_tmp.iterrows():      
        str1=str(getattr(row, 'Labels'))      
        temp_ans=str1.split(',')
        temp_ans.sort(key=lambda x: genres.index(x))
        separator = ","
        result = separator.join(temp_ans)
        data_train.loc[index,'Labels']=result  
    print(data_train['Labels'])
    train=data_train.iloc[1000:]
    dev=data_train.iloc[:1000]
    # dev.to_csv("test.csv", index=False)
    train.to_csv("train_sort.csv", index=False)
    #data_train.to_csv("train.csv",index=False)
before()    
print(genres)  
data_train = pd.read_csv('train_sort.csv')
data_tmp=pd.DataFrame(data_train,columns=['ID', 'Text', 'Labels'])
print(data_train.head(100))
# prompt = """情绪文本分类任务：将一段语句进行分类，一段语句可以属于多个分类，共分为以下八个分类 爱、快乐、焦虑、悲伤、期望、恨、惊讶、愤怒

# 下面是一些范例:

# 性能比 获得 多种 测试 好评 -> 爱
# 第一 眼 就 看出 这 刚 上市 长安 福特 蒙 迪 欧 — 致胜 想 不 实 车 比 网上 照片 和 电视 里 广告 更加 诱人 -> 爱，惊讶

# 请对下述评论进行分类。返回分类结果，多个分类用逗号隔开，无需其它说明和解释。

# xxxxxx ->
# """

def get_prompt(text):
    return prompt.replace('xxxxxx',text)


response, his = model.chat(tokenizer, get_prompt('性能比 获得 多种 测试 好评->'), history=[])
his.append(("５０ 年 遇 大雪 阻断 无数 游子 回家 归途 也 影响 正常 上下班 -> ","Anxiety"))
his.append(("凑巧 那天 下午 没有 开车 而 天空 飘 起 鹅毛大雪 如果 走 着 回家 说不定 就 会 变成 雪人 就 百感交集 时候 辆 熟悉 车子 停 我的 面前 不 等 车 小柯 送 你 一程 吧 -> ","Surprise,Anxiety,Joy"))

his.append(("瞪 大 眼睛 一看 咦 这 不 可可 嘛 什么 时候 买 新车 -> ","Surprise"))
his.append(("自然 不会错 过 这么 好 机会 顺手 拉开 车门 车 内 暖和 空调 让 人 倍 感 温馨 顿时 忘却 车外 寒冷 -> ","Joy,Expect,Love")) 

    
#将上下文整理成与推理时候一致，参照model.chat中的源码~
#model.build_inputs??
# def predict(text,pred):
#     num=0
#     ans=''    
#     for index,row in data_tmp.iterrows():      
#         str11=str(getattr(row, 'Labels'))
#         if(str11==pred):
#             num=num+1
#             ans=str(getattr(row, 'Text'))
#             break
#     his1=copy.deepcopy(his)
#     # response, history1 = model.chat(tokenizer, ans, history=his1)    
#     response, history = model.chat(tokenizer, f"{text} -> ", history=his1,
#     temperature=0.01)
#     return response 
def predict(text,pred):

    his1=copy.deepcopy(his)
    # i=0
    # while i<num:
    #     his1.append((ans[i],pred))
    #     i=i+1
    response, history = model.chat(tokenizer, f"{text} -> ", history=his1,
    temperature=0.01)
    return response 


text='那 人 就 说 那 你 等 我会 司机 说 你 等 下一 班车 吧 那 人 就 不 肯 下 车 司机 就 不 肯 开车 就 这样 僵持 着'
# print(text)
# print(predict(text)) 
df=pd.read_csv("/data/zhangxiaoming/personal/sentiment_analysis/pre_bert_chinese.csv")
data_tmp=pd.DataFrame(df,columns=['ID', 'Text', 'Labels','pred'])
for index,row in data_tmp.iterrows():
        str1=str(getattr(row, 'Labels'))
        str2=str(getattr(row, 'Text'))
        str4=str(getattr(row, 'pred'))
        str3=predict(str2,str4)
        df.loc[index,'pred_chatglm']=str3
df.to_csv("pre_git_glm_lora1.csv",index=False)
        