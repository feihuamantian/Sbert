# Sbert
sentence-bert实现文本检索、语义匹配、FAQ问答
## 1、来源
参考论文：Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
论文下载地址：https://arxiv.org/abs/1908.10084
该论文的相关代码已开源，github链接：sentence-transformers，sentenc-tansformers文档：官方文档
sentenc-tansformers非常好用，封装的很好，使用简单  https://www.sbert.net/docs/pretrained_models.html

## 2、实现能力
### （1）sentence进行encoding
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

query_embedding = model.encode('How big is London')
passage_embedding = model.encode(['London has 9,787,426 inhabitants at the 2011 census',
                                  'London is known for its finacial district'])

print("Similarity:", util.dot_score(query_embedding, passage_embedding))

#### (2)q1,q2共同编码实现分类
![image](https://user-images.githubusercontent.com/28010145/137055565-e982d7d3-c710-4f00-9888-d2c6d8f5e46f.png)
from sentence_transformers import CrossEncoder
model = CrossEncoder('model_name', max_length=512)
scores = model.predict([('Query1', 'Paragraph1'), ('Query1', 'Paragraph2')])

#For Example
scores = model.predict([('How many people live in Berlin?', 'Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.'), 
                        ('How many people live in Berlin?', 'Berlin is well known for its museums.')])
model_name现有的fine-tinue:
cross-encoder/ms-marco-TinyBERT-L-2-v2 - MRR@10 on MS Marco Dev Set: 32.56
cross-encoder/ms-marco-MiniLM-L-2-v2 - MRR@10 on MS Marco Dev Set: 34.85
cross-encoder/ms-marco-MiniLM-L-4-v2 - MRR@10 on MS Marco Dev Set: 37.70
cross-encoder/ms-marco-MiniLM-L-6-v2 - MRR@10 on MS Marco Dev Set: 39.01
cross-encoder/ms-marco-MiniLM-L-12-v2 - MRR@10 on MS Marco Dev Set: 39.02
