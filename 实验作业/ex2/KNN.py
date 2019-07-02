#coding:utf-8
import os
import numpy as np
import operator
'''
①难点：计算欧氏距离并排序，确定k值，这里k=3最优
②图片都处理为数字文本,文本中没有空格
③字典排序，排序后变为[(),()]形式
'''
#价值数据
def Load(dst):
    if not os.path.exists(dst):
        return
    list=os.listdir(dst)
    length=len(list)
    label=[]
    train=[]
    for i in range(length):
        path=dst+"/"+list[i]
        read=open(path)
        temp = []
        for j in range(28):
            line=read.readline()
            for k in range(28):
                bit=(line[k])
                temp.append(bit)
        train.append(temp)

        m=0
        tt = ''
        while(list[i][m]!= '_'):
            tt = tt + list[i][m]
            m = m +1

        label.append(tt)
    train=np.array(train)
    return train,label

def Classifier(train,label,testPath,KK):
    # print(train)
    # print(train.shape)
    # print(label)

    list=os.listdir(testPath)
    length=len(list)
    errorCount= 0
    for i in range(length):
        #数据处理
        path=testPath+"/"+list[i]

        #实际值
        m=0
        ok = ''
        while (list[i][m] != '_'):
            ok =  ok + list[i][m]
            m = m + 1
        # print(ok)
        read=open(path)
        test=[]
        for j in range(28):
            line=read.readline()
            for k in range(28):
                bit=(line[k])
                test.append(bit)
        #计算欧氏距离,不需要遍历，技巧
        m=train.shape[0]
        test=np.tile(test,(m,1))


        #每一个 test 样本都会跟所有的train 样本计算距离差
        sum =[0]*length
        for j in range(length):
            for k in range(28*28):
                sum[j] = sum[j] + (int(train[j][k]) - int(test[j][k]))*(int(train[j][k]) - int(test[j][k]))     # 对应相减

       # print(test)
       # print(test.shape)


        # index = sum.index(min(sum))
        # print(index)
        # print("hhhh")
        # print(sum)
        # print("index")
        # print(label[index])   #yue

        # 找到k个最近距离的下标
        ans = {}
        for j in range(KK):
            index = sum.index(min(sum))
            sum[index] = 10000
            if label[index] in ans.keys():
                ans[label[index]] = ans[label[index]]+1
            else:
                ans[label[index]] = 1

        print(ans.items())
        ans=sorted(ans.items(),key=operator.itemgetter(1),reverse=True)
        print ("实际值=",ok,"预测值=",ans[0][0])
        if ok != ans[0][0]:
            errorCount += 1.0


    print("错误总数：%d" % errorCount)
    print("测试总数：%d" % length)
    print("错误率：%f" % (errorCount / length))

trainPath="F:/program/Machine Learning/dst"
testPath="F:/program/Machine Learning/dst1"

#训练集处理
train,label=Load(trainPath)
#测试集处理
Classifier(train,label,testPath,4)