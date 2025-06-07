import torch
import torch.nn as nn
import torch.nn.functional as F

class MindModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=64,
                 num_capsules=5, dim_capsule=16, num_routing=3):
        super(MindModel, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_dim = embedding_dim
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.num_routing = num_routing

        # 用户嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        # 电影嵌入层
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        # 用户特征处理
        self.user_fc = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 电影特征处理
        self.movie_fc = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 多兴趣提取层
        self.interest_layer = nn.Linear(128, num_capsules * dim_capsule)

        # 预测层
        self.prediction = nn.Sequential(
            nn.Linear(num_capsules, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, user_input, movie_input):
        # 用户嵌入
        user_emb = self.user_embedding(user_input)
        user_emb = self.user_fc(user_emb)

        # 电影嵌入
        movie_emb = self.movie_embedding(movie_input)
        movie_emb = self.movie_fc(movie_emb)

        # 合并特征
        concat = torch.cat([user_emb, movie_emb], dim=1)

        # 多兴趣提取
        interest = self.interest_layer(concat)
        interest = interest.view(-1, self.num_capsules, self.dim_capsule)

        # 动态路由
        b = torch.zeros(interest.size(0), self.num_capsules, self.dim_capsule).to(interest.device)

        for i in range(self.num_routing):
            # 计算耦合系数
            c = F.softmax(b, dim=1)

            # 计算加权和
            s = torch.sum(c * interest, dim=-1, keepdim=True)

            # 非线性激活
            v = self.squash(s)

            # 更新路由权重
            if i < self.num_routing - 1:
                b = b + torch.sum(v * interest, dim=-1, keepdim=True)

        # 兴趣向量
        interest_vectors = v.squeeze(-1)

        # 预测评分
        pred = self.prediction(interest_vectors)

        return pred

    def squash(self, s):
        squared_norm = torch.sum(s ** 2, dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm) / torch.sqrt(squared_norm + 1e-9)
        return scale * s


def load_model(model_path, num_users, num_movies, device):
    model = MindModel(num_users, num_movies)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model