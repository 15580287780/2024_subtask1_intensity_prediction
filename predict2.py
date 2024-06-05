import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split


class AspectBasedSentimentModel(nn.Module):
    def __init__(self):
        super(AspectBasedSentimentModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('chinese-bert-wwm-ext')
        self.model = BertModel.from_pretrained('chinese-bert-wwm-ext')
        self.linear = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=128)
        outputs = self.model(**inputs)
        representation = outputs[1]
        predict = self.linear(representation)

        return predict


def test_model(valence_model, arousal_model, test_data, output_file, batch_size=1):
    model.eval()
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("ID Intensity\n")
        test_loader = DataLoader(test_data, batch_size)
        for batch in test_loader:
            id, sentence, aspects = batch
            id, = id
            sentence, = sentence

            valence_predictions = []
            arousal_predictions = []
            for aspect in aspects:
                aspect, = aspect
                text = f"{aspect}{sentence}"
                valence_predict = valence_model(text)
                arousal_predict = arousal_model(text)

                valence_predictions.append(valence_predict.item())
                arousal_predictions.append(arousal_predict.item())

            # 初始化一个空字符串用于存储所有方面的结果
            all_aspects_str = []
            # 格式化输出字符串
            for aspect, valence, arousal in zip(aspects, valence_predictions, arousal_predictions):
                aspect, = aspect
                aspect_str = f'({aspect},{valence:.2f}#{arousal:.2f})'
                all_aspects_str.append(aspect_str)
            all_aspects_str = ''.join(all_aspects_str)
            line = f"{id} {all_aspects_str}\n"
            file.write(line)


# 加载数据
def load_data_from_file(file_path, train):
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


# 设置全局随机种子
random_seed = 0
set_seed(random_seed)
# 准备数据
print('加载数据')
# test_data = load_data_from_file('data/SIGHAN2024_dimABSA_Validation_Task1_Simplified.txt', train=False)
test_data = load_data_from_file('data/SIGHAN2024_dimABSA_Testing_Task1_Traditional.txt', train=False)

# 创建模型
print('创建模型')
valence_model = torch.load('./model/best_valence_chinese-bert-wwm-ext.pth')
arousal_model = torch.load('./model/best_arousal_chinese-bert-wwm-ext.pth')


# 测试模型
print('模型测试')
# output_file = "task1s.txt"
output_file = "task1t.txt"
test_model(valence_model, arousal_model, test_data, output_file)
