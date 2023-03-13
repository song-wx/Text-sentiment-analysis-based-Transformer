import torch
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
import random

"""数据集处理"""



class MyDataset(Dataset):
    """自定义数据集 句子+对应情感分类"""

    def __init__(self, sentences, sentiments):
        self.sentences = sentences
        self.sentiments = sentiments

    def __getitem__(self, index):
        return self.sentences[index], self.sentiments[index]
        
    def __len__(self):
        return len(self.sentences)


"""数据处理 原始数据格式 PhraseId SentenceId Phrase	Sentiment"""

class glove_embedding():
    """将句子转化为glove对应的序号表示"""
    def __init__(self, datas, glove_dict, test_rate):
        self.word2ID = dict()  ##单词->ID的字典
        datas.sort(key = lambda x:len(x[2].split()))  ##将数据按照词数的升序排列，以减少一个batch里不必要的padding操作   
        self.word_nums = 0 ##数据中出现过的单词数量
        self.embedding_list = list() ##数据中用到glove中的单词的对应的向量 ps:nn.embedding用到的weights列表就是这个

        self.train, self.test = data_split(datas, test_rate=test_rate) ##由于kaggle中的测试集无gt，故从训练集中划出一部分当测试集
        
        self.train_sentiments = [int(data[3]) for data in self.train] ##数据第四列是同一行的句子对应的情感类别
        self.test_sentiments = [int(data[3]) for data in self.test]
        
        self.longest_sentence = 0 ##数据集中，一句话中包含的最多的单词数量

        self.train_sentences_ID = list() ##list中每一项代表句子中依次各单词在embedding_list中对应的ID
        self.test_sentences_ID = list() 

        self.embedding_list.append([0]*50) ## embedding_list第一个是全零向量，代表padding

        
        """构造nn.embedding()需要的weights列表,首先构造单词和ID的映射，在构造ID和词向量的映射"""
        for data in datas:
            sentence = data[2] ##数据第三项是句子
            sentence.upper() ##将字母大写，以免因大小写将同一单词认定为是不同单词
            words = sentence.split() ##将句子分词

            for word in words:
                if word not in self.word2ID:
                    self.word2ID[word] = len(self.word2ID) + 1 ## 将未记录过的word赋予一个对应的ID
                    if word in glove_dict:
                        self.embedding_list.append(glove_dict[word]) ##embedding_list第word_ID个是word对应的词向量
                    else:
                        self.embedding_list.append(50 * [0]) ##若glove中无单词对应的向量，则用全零向量表示该单词
            
        self.word_nums = len(self.embedding_list) ##数据集涉及的单词数量，未含padding元，ID：0
        
        """构造训练集句子所含单词在embeddinglist中单词对应ID的列表"""
        for data in self.train:
            sentence = data[2] ##数据第三项是句子
            sentence.upper() ##将字母大写，以免因大小写将同一单词认定为是不同单词
            words = sentence.split() ##将句子分词
            words_ID = [self.word2ID[word] for word in words] ##单句话中所含单词的ID
            self.longest_sentence = max(self.longest_sentence, len(words_ID)) ##记录句子中所含有最多的单词数
            self.train_sentences_ID.append(words_ID) 
        
        """构造测试集集句子所含单词在embeddinglist中单词对应ID的列表"""
        for data in self.test:
            sentence = data[2] ##数据第三项是句子
            sentence.upper() ##将字母大写，以免因大小写将同一单词认定为是不同单词
            words = sentence.split() ##将句子分词
            words_ID = [self.word2ID[word] for word in words] ##单句话中所含单词的ID
            self.longest_sentence = max(self.longest_sentence, len(words_ID)) ##记录句子中所含有最多的单词数
            self.test_sentences_ID.append(words_ID)

        


def collate_fn(batch_data):
    """调整数据的输出类型"""
    sentence, sentiment = zip(*batch_data) ## zip(*batch_data)将数据打包成（x, y）形式
    sentences = [torch.LongTensor(sent) for sent in sentence]  # 把句子变成Longtensor类型
    padded_sents = pad_sequence(sentences, batch_first=True, padding_value=0)  # 自动padding
    return torch.LongTensor(padded_sents), torch.LongTensor(sentiment)


def get_dataset_loader(sentences, sentiments, batchsize):
    """创建dataloader供训练使用"""
    my_dataset = MyDataset(sentences, sentiments) ##使用处理好的句子矩阵和对应情感类别构造Dataset
    dataloader = DataLoader(dataset=my_dataset, batch_size=batchsize, shuffle=True, drop_last=True, collate_fn=collate_fn)
    return dataloader

def data_split(data, test_rate=0.3):
    """把数据按一定比例划分成训练集和测试集"""
    train = list()
    test = list()
    for datum in data:
        if random.random() > test_rate: ##取0-1之间的随机数，大于0.3则进训练集
            train.append(datum)
        else:
            test.append(datum)
    return train, test

            





