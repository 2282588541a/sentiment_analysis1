#%%
import json
import datasets
from typing import Any, Dict, List


_DESCRIPTION = "An example of dataset."
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""
_URL = "examples.json"


# class ExampleDataset(datasets.GeneratorBasedBuilder):

#     VERSION = datasets.Version("0.0.0")

#     def _info(self) -> datasets.DatasetInfo:
#         features = datasets.Features({
#             "instruction": datasets.Value("string"),
#             "input": datasets.Value("string"),
#             "output": datasets.Value("string"),
#             "history": datasets.Sequence(datasets.Sequence(datasets.Value("string")))
#         })
#         return datasets.DatasetInfo(
#             description=_DESCRIPTION,
#             features=features,
#             homepage=_HOMEPAGE,
#             license=_LICENSE,
#             citation=_CITATION
#         )

#     def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
#         file_path = dl_manager.download(_URL)
#         return [
#             datasets.SplitGenerator(
#                 name=datasets.Split.TRAIN,
#                 gen_kwargs={
#                     "filepath": file_path
#                 }
#             )
#         ]

#     def _generate_examples(self, filepath: str) -> Dict[int, Dict[str, Any]]:
#         example_dataset = json.load(open(filepath, "r", encoding="utf-8"))
#         for key, example in enumerate(example_dataset):
#             yield key, example
history=[
    ['''情绪文本分类任务：将一段语句进行分类，一段语句可以属于多个分类，共分为以下八个分类 Love,Joy,Anxiety,Sorrow,Expect,Hate,Surprise,Anger
请对下述评论进行分类，并说明理由，多个分类用逗号隔开''',"如果您能提供一段评论或语句，我将非常乐意帮助您进行分类并提供相应的理由。"],

        [
                "５０ 年 遇 大雪 阻断 无数 游子 回家 归途 也 影响 正常 上下班",'''这段评论包含以下分类:
Anxiety

理由:

"遇大雪阻断无数游子回家归途"表明大雪天气阻挡了许多离家游子的归途,可能感到担忧无法按时回家
"也影响正常上下班"一语,说明大雪对日常上下班造成影响,可能引发人们工作是否如期的焦虑
总体来说,这段评论着重描述大雪对人们生活规律的干扰,可能使读者有回家或上下班计划面临难题的担忧之情,属于Anxiety情绪分类。'''
        ],
       
        [
         "凑巧 那天 下午 没有 开车 而 天空 飘 起 鹅毛大雪 如果 走 着 回家 说不定 就 会 变成 雪人 就 百感交集 时候 辆 熟悉 车子 停 我的 面前 不 等 车 小柯 送 你 一程 吧 ",'''这段评论包含以下分类:
Surprise, Anxiety, Joy

理由:

"凑巧那天下午没有开车"和"熟悉车子停我的面前",均表示出乎意料的情况,属于Surprise分类;

"如果走着回家说不定就会变成雪人",描述如果没有车可能面临的困境,产生一定的焦虑,属于Anxiety分类;

"小柯送你一程",表示受到助力的喜悦,属于Joy分类。

整体来看,这段评论既包含了惊讶的情绪因素,也带有一定程度的焦虑,但通过别人的帮助得以解脱,最终体现出感激和喜悦。'''
],
       ["瞪 大 眼睛 一看 咦 这 不 可可 嘛 什么 时候 买 新车", '''这段评论包含以下分类:
Surprise

理由:

通过使用"瞪大眼睛一看"这样的表述词,表示看见什么东西感到惊奇。"咦"也是一种惊奇的语气词。"这不可可嘛"进一步确认了看见东西感到意外。整个表达偏重于表示惊奇的情绪。没有其他词语表现出其他情绪如喜悦等。所以该评论仅包含Surprise一个分类。'''
       ],
       
        [
                "自然 不会错 过 这么 好 机会 顺手 拉开 车门 车 内 暖和 空调 让 人 倍 感 温馨 顿时 忘却 车外 寒冷",'''此评论主要表现出Joy、Expect和Love三种情绪。

理由如下:

"顺手拉开车门"、"车内暖和空调"再次表明用户突然间获得了一个舒适温暖的环境,产生的惊喜之情属于Joy分类。

"顿时忘却车外寒冷",暗示用户感到这个状况可带来温暖,体现出一丝期待,属于Expect分类。

此外,描述的语气中也暗含了对这段温馨时光的留恋和珍惜之情,显示出某种亲切感和爱意,属于Love分类。

综上,这段评论主要属于Joy因为获得温暖惊喜时产生的快乐,同时也包含用户对这个温暖环境的期待与留恋之情,表现出Expect和Love两种情绪。所以正确分类为Joy、Expect和Love三类。'''
],
       
        [
                "这 款 绚丽 无比 轿车 但是 她 价格 出乎 我的 意料 如此 高 性价比 却 只 等 车 价格",'''这段评论包含以下分类:
Love,Surprise

理由:

评论表达了对这款车外表设计的赞赏,使用了"绚丽无比"这样的词汇表现喜爱,这体现出Love分类;同时通过"但是她价格出乎我的意料如此高"表示对车价格的惊讶,价格远超预期,这反映了Surprise分类。'''
        ]
       
    ]    
instruction='''情绪文本分类任务：将一段语句进行分类，一段语句可以属于多个分类，共分为以下八个分类 Love,Joy,Anxiety,Sorrow,Expect,Hate,Surprise,Anger
请对下述评论进行分类，并说明理由，多个分类用逗号隔开'''
example={}
example['input']=""
example["history"]=history
import csv
import pandas as pd
import copy
ans=[]
data=pd.read_csv("githubtrain.csv")
data=data.iloc[:2000]#测试集选择前2000条
data_tmp=pd.DataFrame(data,columns=['ID', 'Text', 'cot'])
for index,row in data_tmp.iterrows():
    example1=copy.deepcopy(example)
    example1["instruction"]=str(getattr(row, 'Text'))
    str1=str(getattr(row, 'cot'))
    example1["output"]=str1
    ans.append(example1)
with open("data1.json", "w") as file:
    # 逐个写入数据并换行
    for item in ans:
        json.dump(item, file, ensure_ascii=False)  # 将数据写入文件
        file.write("\n")  # 写入换行符

# 关闭文件
file.close()
   
#%%
