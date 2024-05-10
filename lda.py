import jieba
import os  # 用于处理文件路径
import numpy as np
from gensim import corpora
import pickle
from gensim import models
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score

def content_deal(content):
    ad = '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com'  #去除无意义的广告词
    content = content.replace(ad, '')
    content = content.replace("\u3000", '')
    return content


def read_novel(path,stop_word_list):
    file_list = os.listdir(path)
    word_list = {}
    char_list = {}
    for file in file_list:
            novel_path = r"data/" + file
            char = []
            word = []

            with open(novel_path, 'r', encoding='gb18030') as f:
                content = f.read()
                word_list0 = content_deal(content)
                # 大于500词的段落
                for para in word_list0.split('\n'):
                    if len(para) < 1:
                        continue
                    char.append([char for char in para if char not in stop_word_list and char != ' '])
                    word.append([word for word in jieba.lcut(para) if word not in stop_word_list and word != ' '])
                    file_name = os.path.splitext(file)[0]
                    f.close()
            char_list[file_name] = char
            word_list[file_name] = word

    return char_list, word_list

def dic_matrix(fenci_word,fenci_char):
    dic_word = corpora.Dictionary(fenci_word)
    cor_word = [dic_word.doc2bow(i) for i in fenci_word]
    dic_char = corpora.Dictionary(fenci_char)
    cor_char = [dic_char.doc2bow(i) for i in fenci_char]
    return dic_word,cor_word,dic_char,cor_char

def para_t_freq(lda_word, cor_word):
    para_topic = []
    # 注释部分用于打印高频词，不建议在长文本的情况下使用
    # count = 0
    for i in cor_word:
        # count += 1
        topic = lda_word.get_document_topics(i)
        init = np.zeros(lda_word.num_topics)
        # topic_words = []
        for ii, value in topic:
            init[ii] = value
            # topic_words.append(lda_word.print_topic(ii, topn=10))
        para_topic.append(init)
        # if count < 6:
        #     print(topic_words)  # 打印高频词
    return para_topic


def evenly_sample_data(word_labels, total_samples):
    label_counts = np.bincount(word_labels)  # 统计每个标签出现的次数
    num_labels = len(label_counts)
    # 计算每个标签需要取得的数据数量
    samples_per_label = total_samples // num_labels
    # 初始化结果列表
    sampled_data = []
    # 遍历每个标签，取得相应数量的数据
    for label in range(num_labels):
        label_indices = [index for index, value in enumerate(word_labels) if value == label]
        if len(label_indices) == 0:
            continue
        label_data = np.random.choice(label_indices, samples_per_label, replace=True)  # 从当前标签的数据中随机抽取
        sampled_data.extend(label_data)
    # 如果总数不够，随机从所有数据中取补足数量
    while len(sampled_data) < total_samples:
        random_index = np.random.randint(len(word_labels))
        sampled_data.append(random_index)
    return sampled_data

if __name__ == '__main__':
    path = r"data/"
    stop_word_file = r"cn_stopwords.txt"
    punctuation_file = r"cn_punctuation.txt"
    ll = r"data/"

    """语料库预处理，第一次运行，之后可省略"""
    # # 读取停词列表
    stop_word_list = []
    with open(stop_word_file, 'r', encoding='utf-8') as f:
        for line in f:
            stop_word_list.append(line.strip())
    stop_word_list.extend("\u3000")
    stop_word_list.extend(['～', ' ', '没', '听', '一声', '道', '见', '中', '便', '说', '一个', '说道'])
    # 读取段落
    # 处理前删除文件夹内inf.txt
    # char_dict, word_dict = read_novel(ll,stop_word_list)
    # with open('word_dict.pkl', 'wb') as f:
    #     pickle.dump(word_dict, f)
    # with open('char_dict.pkl', 'wb') as f:
    #     pickle.dump(char_dict, f)
    """语料库预处理，第一次运行，之后可省略"""

    """直接读取保存的数据"""
    with open('word_dict.pkl', 'rb') as f:
        word_dict = pickle.load(f)
    with open('char_dict.pkl', 'rb') as f:
        char_dict = pickle.load(f)
    book_names = ['白马啸西风', '碧血剑', '飞狐外传', '连城诀', '鹿鼎记', '三十三剑客图', '射雕英雄传', '神雕侠侣',
                  '书剑恩仇录',
                  '天龙八部', '侠客行', '笑傲江湖', '雪山飞狐', '倚天屠龙记', '鸳鸯刀', '越女剑']
    book_names_id = {name: i for i, name in enumerate(book_names)}
    # data_label_ids = [book_names_id[label] for label in data_label]

    tokens = 1000  # 修改这两项得到不同的分类效果
    topics = 500
    print("token length:",tokens)
    print("topic num:",topics)
    "词"
    word_corpus = []
    word_labels = []
    for file_name, para in word_dict.items():
        word = []
        for words in para:
            word.extend(words)
            if len(word) > tokens:
                word_corpus.append(word[:tokens])
                word_labels.append(file_name)
                word = []
    word_label_ids = [book_names_id[label] for label in word_labels]
    sample_labels = evenly_sample_data(word_label_ids, 1000)

    word_cor = [word_corpus[i] for i in sample_labels]
    word_label = [word_label_ids[i] for i in sample_labels]
    id2word = corpora.Dictionary(word_cor)
    train_word, test_word, train_word_labels, test_word_labels = train_test_split(word_cor, word_label,
                                                                                  test_size=0.1, random_state=42)
    train_cor = [id2word.doc2bow(text) for text in train_word]
    test_cor = [id2word.doc2bow(text) for text in test_word]
    lda_model = models.ldamodel.LdaModel(corpus=train_cor, num_topics=topics, id2word=id2word,
                                         random_state=42, chunksize=800, passes=10,
                                         alpha='auto', per_word_topics=True, dtype=np.float64)
    train_para_topic = para_t_freq(lda_model, train_cor)
    test_para_topic = para_t_freq(lda_model, test_cor)
    total_para_topic = train_para_topic+test_para_topic
    total_labels = train_word_labels+test_word_labels
    classifier = SVC()
    scores = np.mean(cross_val_score(classifier, total_para_topic, total_labels, cv=10, scoring='accuracy', n_jobs=-1))
    print("word Accuracy:", scores)

    "字"
    char_corpus = []
    char_labels = []
    for file_name, para in char_dict.items():
        char = []
        for chars in para:
            char.extend(chars)
            if len(char) > tokens:
                char_corpus.append(char[:tokens])
                char_labels.append(file_name)
                char = []
    char_label_ids = [book_names_id[label] for label in char_labels]
    sample_labels = evenly_sample_data(char_label_ids, 1000)

    char_cor = [char_corpus[i] for i in sample_labels]
    char_label = [char_label_ids[i] for i in sample_labels]
    id2word = corpora.Dictionary(char_cor)
    train_char, test_char, train_char_labels, test_char_labels = train_test_split(char_cor, char_label,
                                                                                  test_size=0.1, random_state=42)
    train_cor = [id2word.doc2bow(text) for text in train_char]
    test_cor = [id2word.doc2bow(text) for text in test_char]
    lda_model = models.ldamodel.LdaModel(corpus=train_cor, num_topics=topics, id2word=id2word,
                                         random_state=42, chunksize=800, passes=10,
                                         alpha='auto', per_word_topics=True, dtype=np.float64)
    train_para_topic = para_t_freq(lda_model, train_cor)
    test_para_topic = para_t_freq(lda_model, test_cor)
    total_para_topic = train_para_topic+test_para_topic
    total_labels = train_word_labels+test_word_labels
    classifier = SVC()
    scores = np.mean(cross_val_score(classifier, total_para_topic, total_labels, cv=10, scoring='accuracy', n_jobs=-1))
    print("char Accuracy:", scores)