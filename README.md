# sentiment_analysis

目前做的工作主要分成三部分，利用bert加线性层对于情绪进行预测，通过微调chatglm2模型进行预测，以及将两者进行结合，

## 文件描述：

data文件夹里存放的是数据来源，目前只有训练集，因此会对训练集进行拆分成训练集和验证集。

bert文件夹里，test1.ipynb是训练代码，pre_bert_chinese.csv是训练后的结果

## 评价标准：

共4种评价指标，分别是对特征进行one-hot编码后的acc，每一条预测的acc，F1 Score (Micro)，F1 Score (Macro)

## bert：

利用'bert-base-chinese 对语句进行编码后送入全连接层转化为8维预测值，对结果通过one-hot编码也转化成八维值，然后进行训练，共训练15个epoch

结果如下：

第一个acc为![](/image/bert1.png)

后面的结果为：

![](/image/bert2.png)



## chatglm2:

利用chatglm2-6b模型进行微调，参考了torch-koreas的部分代码，训练时间单卡A6000运行5h。为了避免脱靶现象，选择先进行几轮正确对话在进行预测，参考代码如下：

```python
prompt = """情绪文本分类任务：将一段语句进行分类，一段语句可以属于多个分类，共分为以下八个分类 Love,Joy,Anxiety,Sorrow,Expect,Hate,Surprise,Anger

下面是一些范例:

性能比 获得 多种 测试 好评 -> Love
第一 眼 就 看出 这 刚 上市 长安 福特 蒙 迪 欧 — 致胜 想 不 实 车 比 网上 照片 和 电视 里 广告 更加 诱人 -> Love,Surprise

请对下述评论进行分类。只返回分类结果，多个分类用逗号隔开，无需其它说明和解释。

xxxxxx ->
"""
def get_prompt(text):
    return prompt.replace('xxxxxx',text)
response, his = model.chat(tokenizer, get_prompt('性能比 获得 多种 测试 好评->'), history=[])
his.append(("５０ 年 遇 大雪 阻断 无数 游子 回家 归途 也 影响 正常 上下班 -> ","Anxiety"))
his.append(("凑巧 那天 下午 没有 开车 而 天空 飘 起 鹅毛大雪 如果 走 着 回家 说不定 就 会 变成 雪人 就 百感交集 时候 辆 熟悉 车子 停 我的 面前 不 等 车 小柯 送 你 一程 吧 -> ","Surprise,Anxiety,Joy"))

his.append(("瞪 大 眼睛 一看 咦 这 不 可可 嘛 什么 时候 买 新车 -> ","Surprise"))
his.append(("自然 不会错 过 这么 好 机会 顺手 拉开 车门 车 内 暖和 空调 让 人 倍 感 温馨 顿时 忘却 车外 寒冷 -> ","Joy,Expect,Love"))
```

在此基础上再进行对话。

效果如下依次为4个评价指标

![](/image/chatglm1.png)

## bert+chatglm2:

目前正在尝试，已经做了几个版本都是基于利用bert的结果修改chatglm2的hitstory，

第一个：选择将历史中的后两个句子替换成和要预测句子情感标签相同的句子，接下来做了两个尝试，分别是，替换成句子+正确答案的模式，这种经过训练发现模型过拟合严重，产生的结果始终和bert的结果保持一致，第二个，送入句子+模型产生答案，这种训练后发现没什么效果。和glm2微调没什么区别！

目前在尝试第二个想法，做完会继续上传

<<<<<<< HEAD
## llama2-7b：

尝试对于llama27b模型进行微调，使用lora方法进行处理，但脱靶现象很严重，目前来看该项工作还是chatglm2最合适。

## 二分类任务辅助：

经过对于chatglm的结果的评估，发现模型不同分类预测的结果存在差异，结果如下。



于是尝试做

```
Love
Accuracy Score = 0.771
F1 Score (Micro) = 0.771
F1 Score (Macro) = 0.7699045153978088
Joy
Accuracy Score = 0.736
F1 Score (Micro) = 0.736
F1 Score (Macro) = 0.6089468226929343
Anxiety
Accuracy Score = 0.762
F1 Score (Micro) = 0.762
F1 Score (Macro) = 0.6809583047357584
Sorrow
Accuracy Score = 0.8
F1 Score (Micro) = 0.8000000000000002
F1 Score (Macro) = 0.7401247401247402
Expect
Accuracy Score = 0.804
F1 Score (Micro) = 0.804
F1 Score (Macro) = 0.478584729981378
Hate
Accuracy Score = 0.848
F1 Score (Micro) = 0.848
F1 Score (Macro) = 0.6572593374281823
Surprise
Accuracy Score = 0.959
F1 Score (Micro) = 0.959
F1 Score (Macro) = 0.48953547728432867
Anger
Accuracy Score = 0.91
F1 Score (Micro) = 0.91
F1 Score (Macro) = 0.5517124584088782
all
Accuracy Score = 0.237
F1 Score (Micro) = 0.5281124497991968
F1 Score (Macro) = 0.3618506512922561
```

多个二分类任务作为辅助，目前仅用单个二分类任务做测试工作。做了Love的二分类任务，发现效果几乎不变，如下文：

```
Love1
Accuracy Score = 0.769
F1 Score (Micro) = 0.769
F1 Score (Macro) = 0.7687481667535947
all
Accuracy Score = 0.769
F1 Score (Micro) = 0.769
F1 Score (Macro) = 0.7687481667535947
```



## 参考：

1.https://github.com/lyhue1991/torchkeras
2.https://www.kaggle.com/code/debarshichanda/bert-multi-label-text-classification

3.https://github.com/FlagAlpha/Llama2-Chinese
