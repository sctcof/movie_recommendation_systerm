# movie_recommendation_systerm
主要是基于电影的数据进行不同算法模型的测试和实际应用，弥补以往项目算法脱离实际应用的差距。
调试了3天，终于算是完成了 ，能实现的效果还是不错了，完成了mind模型的个性化推荐应用。在windows上的显示不太好，等到后续在进行优化调整。
![image](https://github.com/user-attachments/assets/589ec787-bae2-4836-af38-960456489993)

windows上安装 faiss-cpu,linux和mac上可以安装faiss-gpu，或针对mac的版本
需要用到annoy,一个向量数据库，windows上最好下载下来，在本地安装

后续将逐步完善算法的内容，针对将现有的多种推荐算法都在场景上进行尝试，以提升实战水平。
适用方式如下：
1、将项目下载下来，尽量在mac或linux上系统上运行，这样更改的地方少一些，主要是window上主要是device=‘cpu’
2、运行 run_ml_mind.py文件，读取数据，训练，模型保存，会自动生成文件夹
![image](https://github.com/user-attachments/assets/4ab733bb-ca39-4c3e-a590-cac329be077c)
3、运行 run_ml_mind_model_test.py脚本，实现item向量的存储到faiss向量数据库，供后续的向量找回适用。
4、运行app.py脚本，会拉起服务和界面，主要包含搜索和个性化推荐两部分。
![image](https://github.com/user-attachments/assets/db557fd9-005d-44cb-aaf3-4ab468ad2182)

搜索主要进行文本匹配，后续会逐步增强
个性化推荐，主要是针对人的个性化推荐，需要选择具体的人，然后点击相关按钮即可。








这里需要特别感谢torch-rechub提供了相关算法的支持。
