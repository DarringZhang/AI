import pandas as pd
import numpy as np
import json
# 短评数据
movie_comment_file = ['../data/movie_comment%s.json' %j for j in [ i for i in range(20,220,20)] +[225,250]]
com = []
for f in movie_comment_file:
    lines = open(f, 'rb').readlines()
    com.extend([json.loads(elem.decode("utf-8")) for elem in lines])
data_com = pd.DataFrame(com)
data_com['movie_id'] = data_com['movie_id'].apply(lambda x: int(x[0][5:]))
data_com['content'] = data_com.content.apply(lambda x: x[0].strip())
data_com['people'] = data_com.people.apply(lambda x: x.strip())
data_com['people'] = data_com.people_url.apply(lambda x: x[30:-1])
data_com['useful_num'] = data_com.useful_num.apply(lambda x: int(x))
def regular_nonstar(x):
    if x == 'comment-time':
        return 'allstar00 rating'
    else:
        return x
data_com['star'] = data_com.star.apply(regular_nonstar).apply(lambda x: int(x[7]))
data_com['time'] = pd.to_datetime(data_com.time.apply(lambda x: x[0]))
print('获取的总短评数：' ,data_com.shape[0])


data_com = data_com[~data_com.comment_id.duplicated()]
print('去重后的总短评数：' ,data_com.shape[0])


# 以下代码是将去重后的短评人URL保存给Scrapy进一步爬虫：
#people_url = data_com.people_url.unique().tolist()
#np.savetxt('../douban_movie/bin/people_url.out', people_url, fmt='%s')
#urllist = np.loadtxt('../douban_movie/bin/people_url.out', dtype='|S').tolist()
#len(urllist)  # 共38599个people


#drop掉用于爬虫时候检查爬取质量的URL信息，并且添加了label信息，标示出给出3星及其以上的为“喜欢”，其他为"不喜欢"：
data_com = data_com.drop(['URL','people_url'], axis=1)
data_com['label'] = (data_com.star >=3) *1
data_com.info()


data_com_X = data_com[data_com.movie_id == 1292052]
print('爬取《肖申克的救赎》的短评数：', data_com_X.shape[0])





# 用朴素贝叶斯完成中文文本分类器

#训练数据样本的label非常的不平衡，正样本是负样本的20倍
data_com_X.label.value_counts()
# 1    993
# 0     47
# Name: label, dtype: int64

#复制负样本20遍使得正负样本平衡，并且drop停用词，最后生成训练集：
import warnings

warnings.filterwarnings("ignore")
import jieba  # 分词包

import numpy
import codecs

# 获取停用词列表
def get_custom_stopwords(fpath):
    with open(fpath,encoding='utf-8') as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    return stopwords_list

stopwords = get_custom_stopwords(r'stopwords.txt')

def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            segs = filter(lambda x: len(x) > 1, segs)
            segs = filter(lambda x: x not in stopwords, segs)
            sentences.append((" ".join(segs), category))
        except:
            print(line)
            continue


data_com_X_1 = data_com_X[data_com_X.label == 1]
data_com_X_0 = data_com_X[data_com_X.label == 0]

# 下采样
sentences = []
preprocess_text(data_com_X_1.content.dropna().values.tolist(), sentences, 'like')
n = 0
while n < 20:
    preprocess_text(data_com_X_0.content.dropna().values.tolist(), sentences, 'nlike')
    n += 1

# 生成训练集（乱序）
import random

random.shuffle(sentences)
"""
for sentence in sentences[:10]:
    print(sentence[0], sentence[1])
"""
# 明明 勇敢 心式 狗血 nlike
# 震撼 like


from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_score
#from sklearn.model_selection import train_test_split
x,y=zip(*sentences)

def stratifiedkfold_cv(x,y,clf_class,shuffle=True,n_folds=5,**kwargs):
    stratifiedk_fold = StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle)
    y_pred = y[:]
    for train_index, test_index in stratifiedk_fold:
        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(x_train,y_train)
        y_pred[test_index] = clf.predict(x_test)
    return y_pred

NB = MultinomialNB
print(precision_score(y
                      ,stratifiedkfold_cv(vec.transform(x)
                                          ,np.array(y),NB)
                      , average='macro'))
# 0.910392190906





