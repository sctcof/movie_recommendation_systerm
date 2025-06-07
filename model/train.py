import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from mind_model import MindModel
from data_processing import load_and_preprocess_data
import os
import torch.nn as nn


class MovieDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return {
            'user': torch.tensor(self.users[idx], dtype=torch.long),
            'movie': torch.tensor(self.movies[idx], dtype=torch.long),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float)
        }


def train_model(data_dir='data', model_dir='models', epochs=10):
    # 加载数据
    data = load_and_preprocess_data(data_dir)
    df = data['df']

    # 准备训练数据
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train_users = train['user_id_encoded'].values
    train_movies = train['movie_id_encoded'].values
    train_ratings = train['rating'].values / 5.0  # 归一化

    test_users = test['user_id_encoded'].values
    test_movies = test['movie_id_encoded'].values
    test_ratings = test['rating'].values / 5.0  # 归一化

    # 创建数据集和数据加载器
    train_dataset = MovieDataset(train_users, train_movies, train_ratings)
    test_dataset = MovieDataset(test_users, test_movies, test_ratings)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    model = MindModel(data['num_users'], data['num_movies'])
    model.to(device)

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            users = batch['user'].to(device)
            movies = batch['movie'].to(device)
            ratings = batch['rating'].to(device)

            # 前向传播
            outputs = model(users, movies)
            loss = criterion(outputs, ratings.view(-1, 1))

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * users.size(0)

        # 打印训练损失
        avg_loss = total_loss / len(train_dataset)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

    # 保存模型
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'mind_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f'模型已保存至 {model_path}')

    return model


if __name__ == '__main__':
    train_model(epochs=15)