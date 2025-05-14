英文的自我介绍
```text
你先自我介绍一下？
模板:自己+公司+职位+过往项目+主要优势+谢谢

面试官您好,我叫xxx,我毕业于xxx数学系,
我的第一份工作是在xxx做反洗钱算法工程师,主要是做数据收集和预处理,交易模式分析,异常交易特征分析,数据可视化和报告
在23年来到了xxx,至今已一年有余,期间共参与项目8个,开展培训3场,工作内容主要是在模型差异性分析,数据可视化,以及模型构建和优化

在工作了这些年中,我觉得我有2个个人优势:
其一是热衷于分享:比方github开源项目已经近50个,运营社交媒体账号,有粉丝900余名
第二是:喜欢探索和持续学习:比如自学了大模型相关的知识,自学了日语和吉他,做过自己的网站,并且对AI领域非常感兴趣
以上就是我的自我介绍,谢谢老师.
```
1. 重点讲解一下langchain实际使用,比如QA系统
2. 微调步骤方法,实际的问题,微调效果
3. QA之类的AI系统搭建之后怎么评估的
4. 之前公司模型的搭建优化是在做什么?
5. 注意力机制
6. 服务器购买的一些信息,A卡N卡的一些区别

求求了,让老子过吧


面试过后当天发了一个nlp的挑战给我,内容如下:
```text
# 文件位置:../using_files/zip/atom_nlp_challenge.zip
# 对电影评论进行情感分类
Stage 1:
	Task: Sentiment Classification for movie reviews
	Description: We have 50 movie reviews from different movies, do the sentiment classification for each one of the them and save the result to the new column
	Data: movie_reviews.xlsx
	Expected output: colums = [review_id, review, sentiment]
					rows = 50
# 将描述与评论相关联
Stage 2:
	Task: Relate the description with the review
	Description: Find out which review belongs to which description. the ratio of description/review is not fixed, some with more some with less.
	Data: movie_reviews.xlsx,movie_description.xlsx 
	Expected output: colums = [review_id, description_id]
					rows = 50
# 我是在colab做的,附一下colab的执行记录:
```

Stage 1:

[my_movie_analysis_score.ipynb](../using_files/zip/my_movie_analysis_score.ipynb)

Stage 2:

[my_movie_analysis_description_with_review.ipynb](../using_files/zip/my_movie_analysis_description_with_review.ipynb)



### 20240401问题解析:

增加微调记录

[LLM_Fine_Tuning.ipynb](../2.%B4%F3%C4%A3%D0%CD%D3%C5%BB%AF%BC%BC%CA%F5/fine_tune/LLM_Fine_Tuning.ipynb)

显卡信息:

![img_14.png](../using_files/img/PyTorch/img_14.png)



### Reference
* [显卡](https://juejin.cn/post/7311602485865676810)
