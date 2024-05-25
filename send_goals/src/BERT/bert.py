import pandas as pd
import numpy as np
from transformers import BertForTokenClassification
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch
from tqdm import tqdm
import os
from matplotlib import pyplot as plt

def load_file(path):
    # Load the dataset
    train_sentences = []
    train_labels = []
    print(path)
    with open(path, errors='ignore') as f:
        sentence = ''
        labels = ''
        for line in f:
            line = line.strip()
            if line:
                word, pos, chunk, lab = line.split()
                sentence = sentence + word + ' '
                labels = labels + lab + ' '
            else:
                train_sentences.append(sentence)
                train_labels.append(labels)
                sentence = ''
                labels = ''

    return train_sentences, train_labels


def align_label_example(tokenized_input, labels):
    word_ids = tokenized_input.word_ids()
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            label_ids.append(-100)
            # label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
        previous_word_idx = word_idx
    return label_ids


def align_label(texts, labels):
    # 首先tokenizer输入文本
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)
    # 获取word_ids
    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []
    # 采用上述的第一中方法来调整标签，使得标签与输入数据对其。
    for word_idx in word_ids:
        # 如果token不在word_ids内，则用 “-100” 填充
        if word_idx is None:
            label_ids.append(-100)
        # 如果token在word_ids内，且word_idx不为None，则从labels_to_ids获取label id
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        # 如果token在word_ids内，且word_idx为None
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


# 构建自己的数据集类
class DataSequence(torch.utils.data.Dataset):
    def __init__(self, df):
        # 根据空格拆分labels
        lb = [i.split() for i in df['labels'].values.tolist()]
        # tokenizer 向量化文本
        txt = df['text'].values.tolist()
        self.texts = [tokenizer(str(i),
                                padding='max_length', max_length=512,
                                truncation=True, return_tensors="pt") for i in txt]
        # 对齐标签
        self.labels = [align_label(i, j) for i, j in zip(txt, lb)]

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_data, batch_labels


class BertModel(torch.nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained(
            "/home/aaa/bert-base-uncased",
            # 'bert-base-cased',
            num_labels=len(unique_labels))

    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask,
                           labels=label, return_dict=False)
        return output


def train_loop(model, df_train, df_val):
    print("train_loop")
    # 定义训练和验证集数据
    train_dataset = DataSequence(df_train)
    val_dataset = DataSequence(df_val)
    # 批量获取训练和验证集数据
    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=1)
    # 判断是否使用GPU，如果有，尽量使用，可以加快训练速度
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义优化器
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    if use_cuda:
        model = model.cuda()
    # 开始训练循环
    best_acc = 0
    best_loss = 1000
    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0
        # 训练模型
        model.train()
        # 按批量循环训练模型
        for train_data, train_label in tqdm(train_dataloader):
            # 从train_data中获取mask和input_id
            train_label = train_label[0].to(device)
            mask = train_data['attention_mask'][0].to(device)
            input_id = train_data['input_ids'][0].to(device)
            # 梯度清零！！
            optimizer.zero_grad()
            # 输入模型训练结果：损失及分类概率
            loss, logits = model(input_id, mask, train_label)
            # 过滤掉特殊token及padding的token
            logits_clean = logits[0][train_label != -100]
            label_clean = train_label[train_label != -100]
            # 获取最大概率值
            predictions = logits_clean.argmax(dim=1)
            # 计算准确率
            acc = (predictions == label_clean).float().mean()
            total_acc_train += acc
            total_loss_train += loss.item()
            # 反向传递
            loss.backward()
            # 参数更新
            optimizer.step()
        # 模型评估
        model.eval()

        total_acc_val = 0
        total_loss_val = 0
        for val_data, val_label in val_dataloader:
            # 批量获取验证数据
            val_label = val_label[0].to(device)
            mask = val_data['attention_mask'][0].to(device)
            input_id = val_data['input_ids'][0].to(device)
            # 输出模型预测结果
            loss, logits = model(input_id, mask, val_label)
            # 清楚无效token对应的结果
            logits_clean = logits[0][val_label != -100]
            label_clean = val_label[val_label != -100]
            # 获取概率值最大的预测
            predictions = logits_clean.argmax(dim=1)
            # 计算精度
            acc = (predictions == label_clean).float().mean()
            total_acc_val += acc
            total_loss_val += loss.item()

        Loss_list_Test.append(total_loss_train / len(df_train))
        Accuracy_list_Test.append(total_acc_train / len(df_train))
        Loss_list_Valid.append(total_loss_val / len(df_val))
        Accuracy_list_Valid.append(total_acc_val / len(df_val))

        print(
            f'''Epochs: {epoch_num + 1} | 
                Loss: {total_loss_train / len(df_train): .3f} | 
                Accuracy: {total_acc_train / len(df_train): .3f} |
                Val_Loss: {total_loss_val / len(df_val): .3f} | 
                Accuracy: {total_acc_val / len(df_val): .3f}''')


def evaluate(model, df_test):
    print("evaluate")
    # 定义测试数据
    test_dataset = DataSequence(df_test)
    # 批量获取测试数据
    test_dataloader = DataLoader(test_dataset, num_workers=1, batch_size=1)
    # 使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    total_acc_test = 0.0
    for test_data, test_label in test_dataloader:
        test_label = test_label[0].to(device)
        mask = test_data['attention_mask'][0].to(device)
        input_id = test_data['input_ids'][0].to(device)

        loss, logits = model(input_id, mask, test_label.long())
        logits_clean = logits[0][test_label != -100]
        label_clean = test_label[test_label != -100]
        predictions = logits_clean.argmax(dim=1)
        acc = (predictions == label_clean).float().mean()
        total_acc_test += acc
    val_accuracy = total_acc_test / len(df_test)
    print(f'Test Accuracy: {val_accuracy: .3f}')


def align_word_ids(texts):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)
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
                label_ids.append(1 if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx
    return label_ids


def evaluate_one_text(model, sentence):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    text = tokenizer(sentence, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
    mask = text['attention_mask'][0].unsqueeze(0).to(device)
    input_id = text['input_ids'][0].unsqueeze(0).to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    print(sentence)
    print(prediction_label)


def load_model_test(mod):
    model = torch.load(mod)
    evaluate_one_text(model, 'Take me to that fucking sofa.')
    evaluate_one_text(model, 'I want to lie on that desk and play the computer')
    evaluate_one_text(model,
                      'In the living room, there is an elegant set of furniture, ranging from sofas and coffee tables to bookshelves and carpets, each carefully chosen. A plush leather sofa pairs harmoniously with a carved wooden coffee table, as if narrating the sediment of time and the taste of life. The bookshelf displays a plethora of books, blending classical and modern, adding a touch of scholarly atmosphere to the space. The carpet, delicate and comfortable, offers a soft touch that is truly inviting. The entire space seems to be a microcosm of a home, with each piece of furniture bearing warmth and memories, providing a sense of comfort and tranquility.')

def draw_train():
    x1 = range(0, EPOCHS)
    x2 = range(0, EPOCHS)
    y1 = Accuracy_list_Test
    y2 = Loss_list_Test
    z1 = Accuracy_list_Valid
    z2 = Loss_list_Valid
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-', label='test')
    plt.plot(x1, z1, 'o-', label='valid')
    plt.title('accuracy vs. epoches')
    plt.ylabel('accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-', label='test')
    plt.plot(x2, z2, 'o-', label='valid')
    plt.xlabel('loss vs. epoches')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    plt.savefig("accuracy_loss.jpg")

if __name__ == '__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(torch.cuda.is_available())
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    get_sentences, get_labels = load_file("furniture.txt")
    df = {"text": get_sentences,
          "labels": get_labels}  # 将列表a，b转换成字典
    df = pd.DataFrame(df)  # 将字典转换成为数据框
    # print(df)
    print("数据读取完成")

    # 根据空格拆分标签，并将它们转换为列表
    labels = [i.split() for i in df['labels'].values.tolist()]
    # print(labels)
    # 检查数据集中有多少标签
    unique_labels = set()
    for lb in labels:
        [unique_labels.add(i) for i in lb if i not in unique_labels]
    print("标签类别")
    print(unique_labels)

    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
    print(labels_to_ids)

    text = df['text'].values
    example = text[31]
    print("例句")
    print(example)

    text = df['text'].values.tolist()
    tokenizer = BertTokenizerFast.from_pretrained("/home/aaa/bert-base-uncased")
    #tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    text_tokenized = tokenizer(example, padding='max_length',
                               max_length=512, truncation=True,
                               return_tensors="pt")
    print("decode例句")
    print(tokenizer.decode(text_tokenized.input_ids[0]))

    label = labels[31]
    word_ids = text_tokenized.word_ids()
    print("word_ids")
    print(word_ids)
    print("bert")
    print(tokenizer.convert_ids_to_tokens(text_tokenized["input_ids"][0]))
    label_all_tokens = False
    new_label1 = align_label_example(text_tokenized, label)
    print(new_label1)

    df = df.sample(frac=1)
    pd.set_option('display.max_rows', None)
    print(df)
    df_train, df_val, df_test = np.split(df, [int(.8 * len(df)), int(.9 * len(df))])
    LEARNING_RATE = 0.005
    EPOCHS = 10
    btmodel = BertModel()
    Loss_list_Test = []
    Accuracy_list_Test = []
    Loss_list_Valid = []
    Accuracy_list_Valid = []
    train_loop(btmodel, df_train, df_val)
    evaluate(btmodel, df_test)
    torch.save(btmodel, 'train_furniture2.pth')

    evaluate_one_text(btmodel, 'A full-length mirror hangs on the bedroom door, reflecting the morning sunlight.')
    evaluate_one_text(btmodel, 'Take me to that fucking sofa.')
    evaluate_one_text(btmodel, 'I want to lie on that desk and play the computer')
    evaluate_one_text(btmodel,
                      'In the living room, there is an elegant set of furniture, ranging from sofas and coffee tables to bookshelves and carpets, each carefully chosen. A plush leather sofa pairs harmoniously with a carved wooden coffee table, as if narrating the sediment of time and the taste of life. The bookshelf displays a plethora of books, blending classical and modern, adding a touch of scholarly atmosphere to the space. The carpet, delicate and comfortable, offers a soft touch that is truly inviting. The entire space seems to be a microcosm of a home, with each piece of furniture bearing warmth and memories, providing a sense of comfort and tranquility.')
    draw_train()
    # load_model_test('train_furniture2.pth')
