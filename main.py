# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
import jieba
from typing import List, Dict
from gensim import corpora, models
from sklearn.svm import SVC

# 声明数据文件路径，依次为数据根目录、语料库目录与停止词目录
dataRootPath = './data/'
corpusFilePath = os.path.join(dataRootPath, 'corpus')
# corpusFilePath = os.path.join(dataRootPath, 'shorten_corpus')
stopWordsFilePath = os.path.join(dataRootPath, 'stopwords')

# 实验参数：
docNum = 1000  # 提取的段落数量
modeList = ['char', 'word']  # 划分模式
docLengthMap = {  # 段落长度
    'char': [20, 100, 500, 1000, 3000],  # 以字划分可以多点
    'word': [20, 100, 500, 1000]  # 以词划分不能太多
}
# docLengthList = [20, 100, 500, 1000, 3000]  # 段落长度
topicNumList = [2, 5, 10, 20, 50, 100]  # 话题数量
crossValidationNum = 10  # 交叉验证组数
resultFilePath = './LDAClassfication.xlsx'  # 结果文件路径


# 简单的工具类，为带标签的doc（段落）数据
class LabeledDoc(object):
    def __init__(self, label: str, docData: list):
        self.label = label
        self.docData = docData


class LDAClassification(object):
    # 初始化参数
    def __init__(self, readMode: str, docNum: int, docLength: int, topicNum: int, crossValidationNum: int):
        if readMode not in ['char', 'word']:
            raise ValueError('readMode must be char or word')
        self.readMode = readMode  # 采用单字或词语模式
        self.docNum = docNum  # 提取的段落数量
        self.docLength = docLength  # 提取的每个段落包含的token数量
        self.topicNum = topicNum  # topic数量
        if docNum % crossValidationNum != 0:
            raise ValueError('crossValidationNum must be divisible by crossValidationNum')
        self.crossValidationNum = crossValidationNum  # 交叉验证组数
        self.sampledData: List[LabeledDoc] = []  # 数据集

    # 读取文件，切分成段落，采样得到sampledData
    def readAndSampleData(self, stopWordsPath: str, corpusPath: str) -> None:
        # 读取stop words
        stopWords = []
        for file in os.listdir(stopWordsPath):
            with open(os.path.join(stopWordsPath, file), 'r', encoding='utf-8') as stopWordFile:
                for line in stopWordFile.readlines():
                    stopWords.append(line.strip())  # 去掉回车
        # 读取语料库文本
        txtNameWithDatas = []  # 语料库名-文本内容
        # 对每个文本文件
        for filePath in os.listdir(corpusPath):
            with open(os.path.join(corpusPath, filePath), 'r', encoding='gb18030') as corpusFile:
                # 读取内容并删除冗余
                rawTxt = corpusFile.read()
                rawTxt = rawTxt.replace('----〖新语丝电子文库(www.xys.org)〗', '')
                rawTxt = rawTxt.replace('本书来自www.cr173.com免费txt小说下载站', '')
                txtData = rawTxt.replace('更多更新免费电子书请关注www.cr173.com', '')
                if self.readMode == 'word':
                    txtData = list(jieba.lcut(txtData))
                words = [word for word in txtData if word not in stopWords and not word.isspace()]

                # 切分为多个paragraphs，按照label保存
                words = np.array(words)
                tailLen = len(words) % self.docLength
                if tailLen != 0:
                    words = words[:-tailLen]
                txtData = np.split(words, len(words) // self.docLength)
                txtNameWithDatas.append((filePath.split('.txt')[0], txtData))
        # 按照每个文件的size，确定每个文件中抽取的段落数量
        sizeArray = np.array([len(tup[1]) for tup in txtNameWithDatas])
        numArrayFloat = self.docNum * sizeArray / sizeArray.sum()
        numArrayInt = np.floor(numArrayFloat)
        while numArrayInt.sum() < self.docNum:
            # floor可能导致总和少于所需，给误差最大的补上1
            maxErrorIdx = np.argmax(numArrayFloat - numArrayInt)
            numArrayInt[maxErrorIdx] += 1
        for i in range(len(numArrayInt)):
            nowParagNums = numArrayInt[i]
            nowLabel = txtNameWithDatas[i][0]
            nowDocs = txtNameWithDatas[i][1]
            sampleParagraphIdxArr = np.random.choice(range(len(nowDocs)), size=int(nowParagNums), replace=False)
            self.sampledData.extend([LabeledDoc(nowLabel, nowDocs[paragIdx]) for paragIdx in sampleParagraphIdxArr])

    # KFold法划分TrainSet与TestSet
    def getTrainTestSet(self, iValidation):
        groupSize = len(self.sampledData) // self.crossValidationNum
        startIdx = iValidation * groupSize
        endIdx = startIdx + groupSize
        testSet = self.sampledData[startIdx:endIdx]
        trainSet = self.sampledData[:startIdx] + self.sampledData[endIdx:]
        return trainSet, testSet

    # 训练LDA模型，返回训练dic、训练corpus以及LDA模型
    def trainLDA(self, trainData):
        trainDocData = [data.docData for data in trainData]
        trainDictionary = corpora.Dictionary(trainDocData)
        trainCorpus = [trainDictionary.doc2bow(text) for text in trainDocData]
        return (models.LdaModel(corpus=trainCorpus, id2word=trainDictionary, num_topics=self.topicNum),
                trainDictionary, trainCorpus)

    # 训练分类器，返回训练模型model，同时输出训练精确度
    def trainClassifier(self, ldaModel, corpus, trainData):
        docLabel = [data.label for data in trainData]
        probabilityForDocs = np.array(ldaModel.get_document_topics(corpus, minimum_probability=0.0))[:, :, 1]
        model = SVC(kernel='linear', probability=True)
        model.fit(probabilityForDocs, docLabel)
        trainAccuracy = model.score(probabilityForDocs, docLabel)
        print('Train accuracy is {:.4f}'.format(trainAccuracy))
        return model, trainAccuracy

    # 用测试集TestSet查看训练精确度
    def classifyTest(self, classifyModel, ldaModel, trainDictionary, testSet):
        testDocData = [data.docData for data in testSet]
        testLabel = [data.label for data in testSet]
        testCorpus = [trainDictionary.doc2bow(text) for text in testDocData]
        testProbability = np.array(ldaModel.get_document_topics(testCorpus, minimum_probability=0.0))[:, :, 1]
        testAccuracy = classifyModel.score(testProbability, testLabel)
        print('Test accuracy is {:.4f}'.format(testAccuracy))
        return testAccuracy

    # 实验主函数，KFold方法进行LDA分类训练与测试
    # 输出平均结果
    def LDAClassifyTrainAndTest(self):
        trainAccuracySum = 0
        testAccuracySum = 0
        np.random.shuffle(self.sampledData)  # shuffle保证采样均匀
        for iValidation in range(self.crossValidationNum):
            trainSet, testSet = self.getTrainTestSet(iValidation)
            ldaModel, trainDictionary, corpus = self.trainLDA(trainSet)
            classifier, trainAccuracy = self.trainClassifier(ldaModel, corpus, trainSet)
            testAccuracy = self.classifyTest(classifier, ldaModel, trainDictionary, testSet)
            trainAccuracySum += trainAccuracy
            testAccuracySum += testAccuracy
        return trainAccuracySum / self.crossValidationNum, testAccuracySum / self.crossValidationNum


if __name__ == '__main__':
    resFileWriter = pd.ExcelWriter(resultFilePath, mode='w')
    for mode in modeList:  # 遍历‘字’与‘词’模式
        resultTrainDict: Dict[str, List[float]] = {}  # 用字典记录Train实验结果
        resultTestDict: Dict[str, List[float]] = {}  # 用字典记录Test实验结果
        docLengthList = docLengthMap[mode]
        for docLength in docLengthList:  # 遍历不同的段落长度
            resultTrainDict[str(docLength)] = []
            resultTestDict[str(docLength)] = []
            for topicNum in topicNumList:  # 遍历不同的话题数量
                ldaClassifyModel = LDAClassification(readMode=mode, docNum=docNum, docLength=docLength,
                                                     topicNum=topicNum, crossValidationNum=crossValidationNum)
                ldaClassifyModel.readAndSampleData(stopWordsPath=stopWordsFilePath, corpusPath=corpusFilePath)
                meanTrain, meanTest = ldaClassifyModel.LDAClassifyTrainAndTest()
                print('Mode:{}, docLength:{}, topicNum:{}:'.format(mode, docLength, topicNum))
                print('Average train accuracy:{:.4f}, average test accuracy:{:.4f}'.format(meanTrain, meanTest))
                resultTrainDict[str(docLength)].append(meanTrain)
                resultTestDict[str(docLength)].append(meanTest)
        trainDf = pd.DataFrame(resultTrainDict, index=topicNumList)
        testDf = pd.DataFrame(resultTestDict, index=topicNumList)
        trainDf.to_excel(resFileWriter, sheet_name=mode + 'train', index=True)
        testDf.to_excel(resFileWriter, sheet_name=mode + 'test', index=True)
    resFileWriter.close()