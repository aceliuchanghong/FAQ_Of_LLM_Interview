### FAQ_Of_LLM_Interview

大模型算法岗面试题(含答案):常见问题和概念解析 "大模型面试题"、"算法岗面试"、"面试常见问题"、"大模型算法面试"、"大模型应用基础"
### 欢迎PR


### 目录
```text
FAQ_Of_LLM_Interview/
|
├── LICENSE
├── README.md
├── requirements.txt
├── 面试必问问题.md
├── 1.大模型应用基础/
│   ├── CNN神经网络基础.md
│   ├── PyTorch从0开始编写Transformer.md
│   ├── PyTorch搭建神经网络.md
│   ├── Transformer模型结构.md
│   ├── 大模型的泛化能力.md
│   ├── 聚类.分类_算法.md
│   └── 训练与推理.md
├── 2.大模型优化技术/
│   ├── 常见大模型调用代码.md
│   ├── 微调优化.md
│   └── fine_tune/
│       ├── LLM_Fine_Tuning.ipynb
│       └── LLM_Fine_Tuning.md
├── 3.interview_qa/
│   ├── atom.md
│   ├── pdd.md
│   ├── 任意子智能.md
│   └── 联通.md
└── langchain/
    ├── GPT&Bert.md
    ├── LC&Extract.md
    ├── LangChain&CSV.md
    ├── LangChain&LCEL.md
    ├── LangChain&SQL.md
    ├── LangChain&Server&Cli.md
    ├── LangChain.md
    ├── LoRA..ETC.md
    ├── Pinecone&Faiss&Chroma.md
    ├── Pytorch&DeepSpeed.md
    ├── fine-tune参数解释.md
    └── paddle&tensorflow.md
```
### 来一个测试环境
```shell
pip freeze > requirements.txt
conda create -n myPlot python=3.11
conda activate myPlot
pip install -r requirements.txt --proxy=127.0.0.1:10809
```
### 必备知识
在阅读本文前，建议补充一些相关知识。若你之前未了解过相关原理，可以参考以下的链接：

* [github库](https://github.com/aceliuchanghong/large_scale_models_learning_log)

### 吐槽
```text
大模型这方向真的卷,新paper,新模型疯狂出,东西出的比我读的还快.
```
