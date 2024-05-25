#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from transformers import BertForTokenClassification
from transformers import BertTokenizerFast
import torch

import csv
from sentence_transformers import SentenceTransformer as SBert
from sentence_transformers.util import pytorch_cos_sim
import matplotlib.pyplot as plt
import numpy as np

filename = '/home/ubuntu/Desktop/catkin_turtlebot3/src/send_goals/src/data_clip.csv'  # 文件名
column_name = 'label'  # 要读取的列名

unique_labels = {'I-ROOM', 'B-FUR', 'O', 'I-FUR', 'B-ROOM'}
labels_to_ids = {'B-FUR': 0, 'B-ROOM': 1, 'I-FUR': 2, 'I-ROOM': 3, 'O': 4}
ids_to_labels = {0: 'B-FUR', 1: 'B-ROOM', 2: 'I-FUR', 3: 'I-ROOM', 4: 'O'}


class BertModel(torch.nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained(
            "/home/ubuntu/Desktop/catkin_turtlebot3/src/send_goals/src/BERT/bert-base-uncased",
            # 'bert-base-cased',
            num_labels=len(unique_labels))

    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask,
                           labels=label, return_dict=False)
        return output


class NLU_Node:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('NLU', anonymous=True)
        self.label_all_tokens = False

        self.tokenizer = BertTokenizerFast.from_pretrained(
            '/home/ubuntu/Desktop/catkin_turtlebot3/src/send_goals/src/BERT/bert-base-uncased')

        self.model = torch.load('/home/ubuntu/Desktop/catkin_turtlebot3/src/send_goals/src/BERT/train_furniture2.pth')

        # 订阅语音信息
        self.speech_subscriber = rospy.Subscriber('/speech_topic', String, self.speech_callback)

        # 发布文本信息
        self.text_publisher = rospy.Publisher('/text_topic', String, queue_size=10)

        self.label = self.read_unique_csv_column(filename, column_name)

        self.Sentence_model = SBert('/home/ubuntu/Desktop/catkin_turtlebot3/src/send_goals/src/all-MiniLM-L6-v2')

    def speech_callback(self, speech):
        # 假设这里调用了一个函数将语音转换为文本
        # 这里简单地将收到的语音消息作为文本消息发布
        text = self.nlu_convert(speech.data)
        nlu.text_publisher.publish(text)

    def nlu_convert(self, data):
        words = data.split()
        prediction_label = self.evaluate_one_text(self.model, data)
        print(words)
        print(prediction_label)

        fur_indices = []
        for i, tag in enumerate(prediction_label):
            if tag == 'B-FUR' or tag == 'I-FUR':
                fur_indices.append(i)

        fur_words = [words[i] for i in fur_indices]
        result = " ".join(fur_words)
        print("result:", result)
        if result == "":
            result = self.sentence_bert_handle(data)
        return result

    def align_word_ids(self, texts):
        tokenized_inputs = self.tokenizer(texts, padding='max_length', max_length=512, truncation=True)
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                try:
                    label_ids.append(1)
                except:
                    label_ids.append(-100)
            else:
                try:
                    label_ids.append(1 if self.label_all_tokens else -100)
                except:
                    label_ids.append(-100)
            previous_word_idx = word_idx
        return label_ids

    def evaluate_one_text(self, model, sentence):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:
            model = model.cuda()

        text = self.tokenizer(sentence, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
        mask = text['attention_mask'][0].unsqueeze(0).to(device)
        input_id = text['input_ids'][0].unsqueeze(0).to(device)
        label_ids = torch.Tensor(self.align_word_ids(sentence)).unsqueeze(0).to(device)

        logits = model(input_id, mask, None)
        logits_clean = logits[0][label_ids != -100]

        predictions = logits_clean.argmax(dim=1).tolist()
        prediction_label = [ids_to_labels[i] for i in predictions]

        return prediction_label

    def read_unique_csv_column(self, filename, column_name):
        unique_column_data = set()  # 用于存储唯一的列数据

        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            headers = reader.fieldnames

            if column_name not in headers:
                print(f"Column '{column_name}' not found in the CSV file.")
                return None

            for row in reader:
                value = row[column_name]
                if value not in unique_column_data:  # 如果值还没有出现过，则添加到集合中
                    unique_column_data.add(value)

        return list(unique_column_data)

    def sentence_bert_handle(self, sentence):
        # 使用模型计算句子和单词之间的相似度
        sentence_embedding = self.Sentence_model.encode(sentence, convert_to_tensor=True)
        word_embeddings = self.Sentence_model.encode(self.label, convert_to_tensor=True)

        # 计算余弦相似度
        similarities = pytorch_cos_sim(sentence_embedding, word_embeddings).squeeze().tolist()

        for value1, value2 in zip(self.label, similarities):
            print(value1, value2)

        # 对相似度列表进行排序并取出前10个元素
        sorted_similarities = sorted(similarities, reverse=True)[:10]
        sorted_label = [self.label[i] for i in np.argsort(similarities)[::-1][:10]]

        plt.figure(figsize=(10, 8))

        # 绘制条形图
        plt.bar(sorted_label, sorted_similarities, label='Values 1')

        # 添加图例
        plt.legend()

        # 添加标签和标题
        plt.xlabel('Categories')
        plt.xticks(rotation=30)
        plt.ylabel('Values')
        plt.title('Bar Chart')

        # 显示图形
        plt.show()

        return sorted_label[0]


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(torch.cuda.is_available())
    try:
        nlu = NLU_Node()
        # while not rospy.is_shutdown():
        #     command = input()
        #     nlu.text_publisher.publish(command)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
