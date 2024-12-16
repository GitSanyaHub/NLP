
import numpy as np
from collections import defaultdict

class LSH:
    def __init__(self, dim=100, num_tables=10, num_hashes=10):
        # инициализация параметров
        self.dim = dim  # длина вектора
        self.num_tables = num_tables  # число таблиц
        self.num_hashes = num_hashes  # число функций

        # инициализация таблиц, векторов
        self.hash_tables = [defaultdict(list) for _ in range(num_tables)]
        self.random_vectors = [np.random.randn(num_hashes, dim) for _ in range(num_tables)] # используем нормальное расперделение
        self.vectors = {}

    def hash_vector(self, vector, random_vectors):
        # хэширование
        projections = np.dot(random_vectors, vector)
        return tuple((projections > 0).astype(int))

    def add_vector(self, vector, vector_id):
        # добавление нового вектора в базу
        self.vectors[vector_id] = vector
        for i in range(self.num_tables):
            hash_value = self.hash_vector(vector, self.random_vectors[i])
            self.hash_tables[i][hash_value].append(vector_id)

    def cosine_similarity(self, vec_a, vec_b):
        # расчет косинусного сходства
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(vec_a, vec_b) / (norm_a * norm_b)

    def query(self, query_vector, max_results=10):
        # поиск подходящих векторов
        candidate_ids = set()
        for i in range(self.num_tables):
            hash_value = self.hash_vector(query_vector, self.random_vectors[i])
            candidate_ids.update(self.hash_tables[i].get(hash_value, []))

        # подсчет сходства для каждого релевантного вектора
        similarities = []
        for candidate_id in candidate_ids:
            candidate_vector = self.vectors[candidate_id]
            similarity = self.cosine_similarity(query_vector, candidate_vector)
            similarities.append((candidate_id, similarity))

        # сортируем по убываю релевантности
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        return similarities[:max_results]