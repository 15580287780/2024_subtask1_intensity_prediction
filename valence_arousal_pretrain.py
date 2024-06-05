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


# 训练函数
def train_model(model, train_data, eval_data, batch_size=32, num_epochs=3):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    criterion = nn.L1Loss()  # 使用 L1Loss 作为训练损失函数
    eval_criterion = nn.L1Loss()  # 使用 L1Loss 作为评估损失函数
    num_training_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    # 进度条
    progress_bar = tqdm(range(num_training_steps))

    best_eval_loss = float('inf')  # 正无穷大

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        valence_total_loss = 0.0
        arousal_total_loss = 0.0

        for batch_idx, batch in enumerate(train_dataloader):
            id, text, valence, arousal = batch
            valence_label = valence.view(-1, 1)
            arousal_label = arousal.view(-1, 1)

            valence_predict, arousal_predict = model(text)
            valence_loss = criterion(valence_predict, valence_label)  # 一个batch_size的平均损失
            arousal_loss = criterion(arousal_predict, arousal_label)
            loss = valence_loss + arousal_loss

            valence_total_loss += valence_loss
            arousal_total_loss += arousal_loss
            total_loss += loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            if (batch_idx + 1) % 100 == 0:  # 每50个批次打印一次
                # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']
                avg_valence_loss = valence_total_loss / (batch_idx + 1)  # 一个epoch上的平均损失
                avg_arousal_loss = arousal_total_loss / (batch_idx + 1)
                avg_train_loss = total_loss / (batch_idx + 1)
                # 在评估集上评估模型
                eval_loss = evaluate_model(model, eval_dataloader, eval_criterion)
                print(f"\nEpoch {epoch + 1}/{num_epochs} - Batch {batch_idx + 1}/{len(train_dataloader)} "
                      f"- lr: {current_lr:.2e} - valence Loss: {avg_valence_loss:.4f} - arousal Loss: {avg_arousal_loss:.4f}"
                      f"- Avg Train Loss: {avg_train_loss:.4f} - Avg Eval Loss: {eval_loss:.4f}")

                # 早停策略
                if eval_loss < best_eval_loss:
                    print(f"最佳模型已更新,eval_loss:', {eval_loss:.4f}")
                    best_eval_loss = eval_loss
                    # 保存最佳模型
                    torch.save(model, f"./model/best_chinese-bert-wwm-ext2.pth")

        eval_loss = evaluate_model(model, eval_dataloader, eval_criterion)
        if eval_loss < best_eval_loss:
            print(f"最佳模型已更新,eval_loss:', {eval_loss:.4f}"),
            best_eval_loss = eval_loss
            # 保存最佳模型
            torch.save(model, f"./model/best_chinese-bert-wwm-ext2.pth")


# 评估函数
def evaluate_model(model, dataloader, criterion):
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            id, text, valence, arousal = batch
            valence = valence.view(-1, 1)
            arousal = arousal.view(-1, 1)

            valence_predict, arousal_predict = model(text)
            valence_loss = criterion(valence_predict, valence)
            arousal_loss = criterion(arousal_predict, arousal)

            eval_loss += valence_loss + arousal_loss

    avg_eval_loss = eval_loss / len(dataloader)
    model.train()  # 将模型设置为训练模式
    return avg_eval_loss


# 加载数据
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


# 设置全局随机种子
random_seed = 0
set_seed(random_seed)
# 准备数据
print('加载数据')
all_data = load_data_from_file('data/train1.txt', train=True)
train_data, eval_data = train_test_split(all_data, test_size=0.1, random_state=random_seed, shuffle=True)

# 创建模型
print('创建模型')
model = AspectBasedSentimentModel()

# 训练模型
print('模型训练')
train_model(model, train_data, eval_data)



