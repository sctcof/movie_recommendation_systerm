import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
print(os.getcwd())

def load_and_preprocess_data(data_dir='data'):
    # 加载MovieLens 1M数据集
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.dat'),
                          sep='::', engine='python',
                          names=['user_id', 'movie_id', 'rating', 'timestamp'])
    movies = pd.read_csv(os.path.join(data_dir, 'movies.dat'),
                         sep='::', engine='python',
                         names=['movie_id', 'title', 'genres'], encoding='latin-1')
    users = pd.read_csv(os.path.join(data_dir, 'users.dat'),
                        sep='::', engine='python',
                        names=['user_id', 'gender', 'age', 'occupation', 'zip_code'])

    # 合并数据
    df = pd.merge(ratings, movies, on='movie_id')
    df = pd.merge(df, users, on='user_id')

    # 特征工程
    df['year'] = df['title'].str.extract(r'\((\d{4})\)').astype(float)
    df['title_clean'] = df['title'].str.replace(r'\(\d{4}\)', '').str.strip()

    # 对用户和电影ID进行编码
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    df['user_id_encoded'] = user_encoder.fit_transform(df['user_id'])
    df['movie_id_encoded'] = movie_encoder.fit_transform(df['movie_id'])

    # 创建用户特征矩阵
    user_features = df[['user_id_encoded', 'gender', 'age', 'occupation']].drop_duplicates()
    user_features.set_index('user_id_encoded', inplace=True)

    # 创建电影特征矩阵
    movie_features = df[['movie_id_encoded', 'title_clean', 'genres', 'year']].drop_duplicates()
    movie_features.set_index('movie_id_encoded', inplace=True)

    # 创建ID到标题的映射
    movie_id_to_title = dict(zip(movie_features.index, movie_features['title_clean']))
    movie_id_to_genres = dict(zip(movie_features.index, movie_features['genres']))

    # 创建标题到ID的映射（用于搜索）
    title_to_movie_id = {title: idx for idx, title in movie_id_to_title.items()}

    return {
        'df': df,
        'user_features': user_features,
        'movie_features': movie_features,
        'user_encoder': user_encoder,
        'movie_encoder': movie_encoder,
        'movie_id_to_title': movie_id_to_title,
        'movie_id_to_genres': movie_id_to_genres,
        'title_to_movie_id': title_to_movie_id,
        'num_users': len(user_encoder.classes_),
        'num_movies': len(movie_encoder.classes_)
    }


# data_df=load_and_preprocess_data(data_dir=os.getcwd())
# print(type(data_df))
# print(data_df['df'].info())
# print(data_df['user_encoder'])