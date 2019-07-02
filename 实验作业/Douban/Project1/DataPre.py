import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import jieba   # 分词包

file = open(r"review.txt", 'r', encoding='utf-8')
reviews = file.readlines()
data = []
for r in reviews:
    data.append([r[0], r[2:]])
d1 = pd.DataFrame(data)
pd.set_option('max_colwidth', 200)

d1.columns = ['sentiment', 'comment']  # 修改列名， 评价1/0    +  文本
# 打印 数据 转换为DataFrame结构后的内容
#print('修改列名后的数据（只显示前5行）：\n' + str(d1.head()))
#print(d1.shape)


# 清洗数据,通过jieba分词
def word_clean(mytext):
    return ' '.join(jieba.lcut(mytext))


x = d1[['comment']]
x['cutted_comment'] = x.comment.apply(word_clean)
print(x.shape)
# 查看分词后的结果
print('数据分词后的结果：\n' + str(x.cutted_comment[:5]))
y = d1.sentiment
print(y.shape)
print(x.head())




# 将数据集拆开为测试集和训练集
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)
print('训练集：'+str(x_train.shape)+' '+str(y_train.shape))
print('测试集：'+str(x_test.shape)+' '+str(y_test.shape))

#print('hhhhhhhhh'+x_train)

# 获取停用词列表
def get_custom_stopwords(fpath):
    with open(fpath,encoding='utf-8') as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    return stopwords_list

stopwords = get_custom_stopwords(r'stopwords.txt')

#对比不去停用词和去掉停用词后矩阵的特征数量的变化：

#不去停用词
from sklearn.feature_extraction.text import CountVectorizer
vect=CountVectorizer()
term_matrix=pd.DataFrame(vect.fit_transform(x_train.cutted_comment).toarray(),columns=vect.get_feature_names())
print('原始的特征数量：'+str(term_matrix.shape))


#去除停用词
vect = CountVectorizer(stop_words=frozenset(stopwords))
term_matrix = pd.DataFrame(vect.fit_transform(x_train.cutted_comment).toarray(), columns=vect.get_feature_names())
print('去掉停用词的特征数量：'+str(term_matrix.shape))


max_df=0.8  # 在超过这一比例的文档中出现的关键词（过于平凡），去除掉。
min_df=3    # 在低于这一数量的文档中出现的关键词（过于独特），去除掉。
vect=CountVectorizer(max_df=max_df,min_df=min_df,stop_words=frozenset(stopwords))
term_matrix = pd.DataFrame(vect.fit_transform(x_train.cutted_comment).toarray(), columns=vect.get_feature_names())
print('进一步处理后的特征数量：'+str(term_matrix.shape))






# 使用贝叶斯预测分类
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

# 利用管道顺序连接工作
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(vect, nb)

# 交叉验证的准确率

from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import train_test_split

cross_result = cross_val_score(pipe, x_train.cutted_comment, y_train, cv=5, scoring='accuracy').mean()
#cross_result = train_test_split(x_train.cutted_comment, y_train, test_size=0.4, random_state=0)
print('交叉验证的准确率：' + str(cross_result))

# 进行预测
pipe.fit(x_train.cutted_comment, y_train)
y_pred = pipe.predict(x_test.cutted_comment)

# python测量工具集
from sklearn import metrics

# 准确率测试
accuracy = metrics.accuracy_score(y_test, y_pred)
print('准确率：' + str(accuracy))
# 混淆矩阵
print('混淆矩阵：' + str(metrics.confusion_matrix(y_test, y_pred)))