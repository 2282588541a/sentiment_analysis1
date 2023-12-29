# sentiment_analysis

~~目前做的工作主要分成三部分，利用bert加线性层对于情绪进行预测，通过微调chatglm2模型进行预测，以及将两者进行结合，~~

最终结果直接点击下面的getstarted，利用大模型生成知识辅助小模型进行情感分类，这相当于前面几个工作的融合，可以看完最后的工作再看先期的过程

- [Getting Started](#getting-started)



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

尝试将二分类与多分类模型进行融合，发现手动配置效果较差，几乎不会提升，效果如下：

```
Love1
Accuracy Score = 0.768
F1 Score (Micro) = 0.768
F1 Score (Macro) = 0.7671617824167001
all
Accuracy Score = 0.768
F1 Score (Micro) = 0.768
F1 Score (Macro) = 0.7671617824167001
```

接下来尝试通过线性层进行转化连接

## Getting Started

## 将LLM作为外部知识库来帮助bert类的小模型来处理：  

我们发现传统的bert类小模型因为参数等原因本身不具有涌现能力，因此，考虑借助大语言模型来对数据集产生注释来帮助小模型获得正确的答案，这里存在一个问题：在不知道正确答案的情况下，如何保证大语言模型的注释是可信的？

因此，我们按照以下格式设计了系统：该系统分成三部分，数据预处理、LLM微调、训练bert分类器。

![](/image/structure.png)

#### 数据预处理：

这里选择了阿里云的qwen-max，通过调用官方api实现访问，选择这个模型主要因为第一，这个模型是免费的，目前参数量较大的模型部署起来很耗费GPU资源。第二，这个模型的参数量很大，官方宣称通义千问2.1（qwen-max千亿级别模型），在分析语句的时候感觉效果还不错，这里设计两套prompt，都是few-shot相关代码见qwen文件夹，其中call_with_messages对应第一个prompt，call_with_messages2对应第二个prompt.利用qwen大模型完成了2万多条数据的注释，见qwen文件夹githubtrain.csv

第一套：

```
        {
             "role": "system",
            "content": '''情绪文本分类任务：将一段语句进行分类，一段语句可以属于多个分类，共分为以下八个分类 Love,Joy,Anxiety,Sorrow,Expect,Hate,Surprise,Anger
请对下述评论进行分类，并说明理由，多个分类用逗号隔开''',
        },
        {
                "role": "user",
                "content": "５０ 年 遇 大雪 阻断 无数 游子 回家 归途 也 影响 正常 上下班",
        },
        {
                "role": "assistant",
                "content": '''这段评论包含以下分类:
Anxiety

理由:

"遇大雪阻断无数游子回家归途"表明大雪天气阻挡了许多离家游子的归途,可能感到担忧无法按时回家
"也影响正常上下班"一语,说明大雪对日常上下班造成影响,可能引发人们工作是否如期的焦虑
总体来说,这段评论着重描述大雪对人们生活规律的干扰,可能使读者有回家或上下班计划面临难题的担忧之情,属于Anxiety情绪分类。''',
        }
```

第二套：

```
        {
             "role": "system",
            "content": '''情绪文本分类任务：将一段语句进行分类，一段语句可以属于多个分类，共分为以下八个分类 Love,Joy,Anxiety,Sorrow,Expect,Hate,Surprise,Anger
请根据答案对下述评论进行分类，并说明理由，多个分类用逗号隔开''',
        },
         {
                "role": "user",
                "content": "５０ 年 遇 大雪 阻断 无数 游子 回家 归途 也 影响 正常 上下班 ,答案为Anxiety",
        },
        {
                "role": "assistant",
                "content": '''这段评论包含以下分类:
Anxiety

理由:

"遇大雪阻断无数游子回家归途"表明大雪天气阻挡了许多离家游子的归途,可能感到担忧无法按时回家
"也影响正常上下班"一语,说明大雪对日常上下班造成影响,可能引发人们工作是否如期的焦虑
总体来说,这段评论着重描述大雪对人们生活规律的干扰,可能使读者有回家或上下班计划面临难题的担忧之情,属于Anxiety情绪分类。''',
        }
```

对于训练集来说，先使用第一套prompt，如果生成的回复与答案不符，则使用第二套prompt更正回复，第二套prompt会产生正确的答案。因此，训练集生成的知识是正确的，是可信的。

#### LLM微调：

这里选择chatglm3-6b，该模型效果较好，且GPU硬件可以满足该模型lora微调所需要的条件。

这里作微调的原因主要是：直接将测试集在没有正确答案的情况下送入LLM产生的结果肯定是不可信的。因此，将上一阶段训练集产生的可信知识作为训练集进行模型微调，在微调后的模型送入测试集，来保证测试集知识的可信性。（可能有疑问，测试集通过微调后的LLM产生的知识一定是可信的吗，下面会有解释）这部分主要参考LLama Factory的工作，通过LLama Factory进行设计，这里只上传对于csv转成json的相关代码，最终的测试集微调结果（data前2000条作为测试集）和lora微调后的权重

#### 训练bert分类器：

这里总计有八个情绪，因此将多标签预测转化为多个单标签预测。

选择将原文和刚才获得的知识拼接后送入bert分类器进行二分类任务。我之前做的bert都是直接进行多分类任务，后来发现将多分类转为单个二分类效果更好，这部分也参考了一个github项目，在下面的参考项目中会列出

## 知识的可信性说明：

经过微调的LLM肯定不可能在测试集上全部做出正确的预测，但我们认为模型结果是错误的，但过程不一定是错误的。只要全局来看，原文+正确率较高的注释会比原文更有内容。这里举例说明，label是正确的标签，predict是微调后的大语言模型的输出，在送入bert之前，只保留理由的部分内容（对于情绪的关键字全部去除），去除后的内容为temp**，我认为知识具有一个相关度与可信性的矛盾，我们可以通过降低知识和最终标签的相关性，来提升它的可信度。**

```
{"label": "这段评论包含以下分类:\nJoy,Expect\n\n理由:\n\n通过使用\"可以断言成批中国科学家专家教授艺术家CEO出现杰出华人议员校长州长乃至总统产生\",表明作者对中国未来在世界舞台上发挥更大作用充满期待和乐观态度,这属于Expect分类。\n\n同时,\"就中国人对世界的贡献\"表达了作者对于中国人才能为世界作出贡献的自豪与喜悦之情,这体现了Joy分类。", 
"predict": "这段评论包含以下分类:\nLove\n\n理由:\n\n\"可以断言成批中国中国人科学家专家教授艺术家CEO出现杰出华人议员校长州长乃至总统产生就中国人对世界贡献\"这句话中，作者表达了对我国科学家、专家、教授、艺术家、CEO以及杰出华人议员、校长、州长等在世界上做出贡献的赞美。这种赞美和肯定体现了对他们的喜爱和敬仰之情，属于Love（爱）情绪分类。"}

```

```
temp："可以断言成批中国中国人科学家专家教授艺术家CEO出现杰出华人议员校长州长乃至总统产生就中国人对世界贡献\"这句话中，作者表达了对我国科学家、专家、教授、艺术家、CEO以及杰出华人议员、校长、州长等在世界上做出贡献的赞美。这种赞美和肯定体现了对他们的喜爱和敬仰之情，属于（爱）情绪分类。
```
## 其它
对于NLP大作业，主要完成了对于chatglm3的指令微调和对bert的分类微调。利用大模型生成外部知识库来辅助小模型做分类。利用qwen大模型完成了2万多条数据的注释，见qwen文件夹githubtrain.csv
对于该方案，目前实验效果来看，该方法还是存在问题的，最近读了几篇检索增强生成(Retrieval Augmented Generation，RAG)的论文，感觉不应该让模型生成不可信知识，应该考虑让他生成知识，再通过RAG找到相关的例子，这个方案目前还在构想，参考以下结构图。


## 参考：

1.https://github.com/lyhue1991/torchkeras
2.https://www.kaggle.com/code/debarshichanda/bert-multi-label-text-classification

3.https://github.com/FlagAlpha/Llama2-Chinese

4.[murray-z/multi_label_classification: 基于pytorch + bert的多标签文本分类（multi label text classification） (github.com)](https://github.com/murray-z/multi_label_classification)
