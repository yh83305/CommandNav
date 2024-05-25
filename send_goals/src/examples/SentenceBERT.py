import csv
from sentence_transformers import SentenceTransformer as SBert
from sentence_transformers.util import pytorch_cos_sim
import matplotlib.pyplot as plt
import numpy as np


def read_unique_csv_column(filename, column_name):
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


# Example usage:
filename = '../data_clip.csv'  # 文件名
column_name = 'label'  # 要读取的列名

if __name__ == '__main__':

    unique_column_data = read_unique_csv_column(filename, column_name)
    if unique_column_data is not None:
        print(f"The unique data in column '{column_name}' is:")
        print(unique_column_data)

    model = SBert('all-MiniLM-L6-v2')

    # 对句子进行编码
    sentence_str = 'where can i sit'
    sentence = [sentence_str]

    # 使用模型计算句子和单词之间的相似度
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    word_embeddings = model.encode(unique_column_data, convert_to_tensor=True)

    # 计算余弦相似度
    similarities = pytorch_cos_sim(sentence_embedding, word_embeddings).squeeze().tolist()

    for value1, value2 in zip(unique_column_data, similarities):
        print(value1, value2)

    # 对相似度列表进行排序并取出前10个元素
    sorted_similarities = sorted(similarities, reverse=True)[:10]
    sorted_unique_column_data = [unique_column_data[i] for i in np.argsort(similarities)[::-1][:10]]

    plt.figure(figsize=(8, 6))

    # 绘制条形图
    plt.bar(sorted_unique_column_data, sorted_similarities, label='similarities')

    # 添加图例
    plt.legend()

    # 添加标签和标题
    plt.xlabel('Categories')
    plt.xticks(rotation=30)
    plt.ylabel('Values')
    plt.title(sentence_str)

    # 显示图形
    plt.show()
