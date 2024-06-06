import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup, set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr


class AspectBasedSentimentModel(nn.Module):
    def __init__(self):
        super(AspectBasedSentimentModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('chinese-bert-wwm-ext')
        self.model = BertModel.from_pretrained('chinese-bert-wwm-ext')
        self.valence_linear = nn.Linear(self.model.config.hidden_size, 1)
        self.arousal_linear = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=128)
        outputs = self.model(**inputs)
        representation = outputs[1]
        valence_predict = self.valence_linear(representation)
        arousal_predict = self.arousal_linear(representation)

        return valence_predict, arousal_predict


def load_data_from_file(file_path, train=True):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()[1:]  # Skip header
        for line in lines:
            line = line.strip().split(', ')
            if train and len(line) == 5:
                id, sentence, aspect, valence, arousal = line
                valence = float(valence)
                arousal = float(arousal)
                text = f"{aspect}{sentence}"
                data.append((id, text, valence, arousal))
            elif not train and len(line) == 3:
                id, sentence, aspect = line
                aspects = aspect.split('#')
                data.append((id, sentence, aspects))
    return data


def calculate_metrics(preds, labels):
    preds = torch.tensor(preds).squeeze().numpy()
    labels = torch.tensor(labels).squeeze().numpy()
    mae = mean_absolute_error(labels, preds)
    pcc, _ = pearsonr(labels, preds)
    return mae, pcc

def get_predictions(model, dataloader):
    model.eval()
    all_valence_preds = []
    all_arousal_preds = []
    all_valence_labels = []
    all_arousal_labels = []

    with torch.no_grad():
        for batch in dataloader:
            id, text, valence, arousal = batch
            valence = valence.view(-1, 1)
            arousal = arousal.view(-1, 1)

            valence_predict, arousal_predict = model(text)

            all_valence_preds.extend(valence_predict.tolist())
            all_arousal_preds.extend(arousal_predict.tolist())
            all_valence_labels.extend(valence.tolist())
            all_arousal_labels.extend(arousal.tolist())

    return all_valence_preds, all_arousal_preds, all_valence_labels, all_arousal_labels


random_seed = 0
set_seed(random_seed)

all_data = load_data_from_file('data/train2.txt', train=True)
train_data, eval_data = train_test_split(all_data, test_size=0.1, random_state=random_seed, shuffle=True)

train_dataloader = DataLoader(train_data, batch_size=32)
eval_dataloader = DataLoader(eval_data, batch_size=32)

best_model = torch.load('./model/best_chinese-bert-wwm-ext2.pth')

eval_valence_preds, eval_arousal_preds, eval_valence_labels, eval_arousal_labels = (
    get_predictions(best_model, eval_dataloader))

eval_valence_mae, eval_valence_pcc = calculate_metrics(eval_valence_preds, eval_valence_labels)
eval_arousal_mae, eval_arousal_pcc = calculate_metrics(eval_arousal_preds, eval_arousal_labels)

print(f'Valence MAE: {eval_valence_mae:.4f}, Valence PCC: {eval_valence_pcc:.4f}')
print(f'Arousal MAE: {eval_arousal_mae:.4f}, Arousal PCC: {eval_arousal_pcc:.4f}')