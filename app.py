import os

import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from data_processing import load_and_preprocess_data
# from model.mind_model import load_model
from model.utils.common import load_mind_model
from model.utils.common import get_user_ebebdding_by_user_id,query_by_embedding
# 初始化应用
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据和模型
print("加载数据...")
data = load_and_preprocess_data('data')
print("加载模型...")
# model = load_model('models/mind_model.pth', data['num_users'], data['num_movies'], device)
model = load_mind_model(os.getcwd())

class RecommendationSystem:
    def __init__(self, model, data, device):
        self.model = model
        self.data = data
        self.device = device

    def recommend_for_user(self, user_id, top_n=10):
        # 获取用户所有评分过的电影
        user_ratings = self.data['df'][self.data['df']['user_id_encoded'] == user_id]
        rated_movies = user_ratings['movie_id_encoded'].unique()

        # 获取用户未评分的电影
        all_movies = self.data['df']['movie_id_encoded'].unique()
        unrated_movies = np.setdiff1d(all_movies, rated_movies)

        if len(unrated_movies) == 0:
            return []

        # 为未评分的电影生成预测
        # user_ids = np.full(len(unrated_movies), user_id)

        # 创建PyTorch张量
        # user_tensor = torch.tensor(user_ids, dtype=torch.long).to(self.device)
        # movie_tensor = torch.tensor(unrated_movies, dtype=torch.long).to(self.device)
        #
        # # 预测评分
        # self.model.eval()
        # with torch.no_grad():
        #     predictions = self.model(user_tensor, movie_tensor).cpu().numpy().flatten()

        # # 获取推荐电影
        # recommended_indices = np.argsort(predictions)[::-1][:top_n]
        # recommended_movies = unrated_movies[recommended_indices]
        # pred_ratings = predictions[recommended_indices]

        users_embedding = get_user_ebebdding_by_user_id(user_id, model, data_path=os.getcwd())
        distence_list, movie_id_list = query_by_embedding(users_embedding, top_n)

        # 创建推荐结果
        recommendations = []
        for i, movie_id in enumerate(movie_id_list):
            title = self.data['movie_id_to_title'].get(movie_id, "未知电影")
            genres = self.data['movie_id_to_genres'].get(movie_id, "未知类型")
            recommendations.append({
                'rank': i + 1,
                'movie_id': int(movie_id),
                'title': title,
                'genres': genres,
                'predicted_rating': 5-float(abs(distence_list[i]))  # 转换为0-5的评分
            })

        return recommendations

    def search_movies(self, query, top_n=20):
        # 简单搜索实现
        results = []
        for movie_id, title in self.data['movie_id_to_title'].items():
            if query.lower() in title.lower():
                genres = self.data['movie_id_to_genres'].get(movie_id, "未知类型")
                results.append({
                    'movie_id': int(movie_id),
                    'title': title,
                    'genres': genres
                })
                if len(results) >= top_n:
                    break
        return results

    def get_user_info(self, user_id):
        # 获取用户信息
        user_row = self.data['user_features'].loc[user_id]
        return {
            'user_id': int(user_id),
            'gender': user_row['gender'],
            'age': int(user_row['age']),
            'occupation': int(user_row['occupation'])
        }


# 创建推荐系统实例
recommender = RecommendationSystem(model, data, device)


@app.route('/')
def index():
    # 随机选择一些用户用于展示
    # sample_df = dataframe.sample(20) if dataframe.shape[0] > 20 else dataframe
    sample_users_df = data['user_features'].sample(10) if data['user_features'].shape[0] > 10 else data['user_features']
    sample_users = sample_users_df.index.tolist()
    user_info = []
    for user_id in sample_users:
        user_info.append({
            'id': int(user_id),
            'gender': data['user_features'].loc[user_id]['gender'],
            'age': int(data['user_features'].loc[user_id]['age'])
        })

    # 获取最受欢迎的电影
    top_movies = data['df']['title_clean'].value_counts().head(10).reset_index()
    top_movies.columns = ['title', 'count']
    top_movies = top_movies.to_dict('records')

    return render_template('index.html', users=user_info, top_movies=top_movies)


@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    top_n = int(request.form.get('top_n', 10))

    # 获取用户信息
    user_info = recommender.get_user_info(user_id)

    # 获取推荐
    recommendations = recommender.recommend_for_user(user_id, top_n)

    return render_template('recommendations.html',
                           user_info=user_info,
                           recommendations=recommendations)


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    if not query:
        return jsonify([])

    results = recommender.search_movies(query)
    return jsonify(results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)