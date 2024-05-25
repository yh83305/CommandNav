from transformers import BertForTokenClassification
from transformers import BertTokenizerFast
import torch

#unique_labels = {'I-ROOM', 'B-FUR', 'O', 'I-FUR', 'B-ROOM'}
#labels_to_ids = {'B-FUR': 0, 'B-ROOM': 1, 'I-FUR': 2, 'I-ROOM': 3, 'O': 4}



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

    return prediction_label


if __name__ == '__main__':
    ids_to_labels = {0: 'B-FUR', 1: 'B-ROOM', 2: 'I-FUR', 3: 'I-ROOM', 4: 'O'}
    label_all_tokens = False
    tokenizer = BertTokenizerFast.from_pretrained('/home/ubuntu/Desktop/catkin_turtlebot3/src/send_goals/src/BERT/bert-base-uncased')
    model = torch.load('/home/ubuntu/Desktop/catkin_turtlebot3/src/send_goals/src/BERT/train_furniture2.pth')
    prediction_label = evaluate_one_text(model, 'I want to sleep')