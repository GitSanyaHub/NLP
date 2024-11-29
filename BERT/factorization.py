import torch.nn as nn
import torch

class FactorizedEmbedding(nn.Module):
    def __init__(self, pretrained_embedding, embedding_dim):
        """
        Факторизованный слой эмбеддингов с инициализацией SVD.

        Args:
            pretrained_embedding (torch.Tensor): Предобученная матрица эмбеддингов (V x H).
            embedding_dim (int): Размер промежуточного пространства эмбеддингов (E).
        """
        super(FactorizedEmbedding, self).__init__()
        vocab_size, original_hidden_dim = pretrained_embedding.size()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Вычисляем SVD разложение
        u, s, vh = torch.linalg.svd(pretrained_embedding, full_matrices=False)

        # Урезаем до embedding_dim
        u = u[:, :embedding_dim]  # (V x E)
        s = s[:embedding_dim]  # (E,)
        vh = vh[:embedding_dim, :]  # (E x H)

        # Создаем матрицу токен-эмбеддингов (V x E)
        self.token_to_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.token_to_embedding.weight.data = u @ torch.diag(s)

        # Создаем линейное преобразование в скрытое пространство (E x H)
        self.embedding_to_hidden = nn.Linear(embedding_dim, original_hidden_dim)
        self.embedding_to_hidden.weight.data = vh.T  # Транспонируем для корректной матрицы

    def forward(self, input_ids):
        """
        Прямой проход через факторизованный слой эмбеддингов.

        Args:
            input_ids (torch.Tensor): Входные индексы токенов.

        Returns:
            torch.Tensor: Скрытые представления (batch_size x seq_length x H).
        """
        embeddings = self.token_to_embedding(input_ids)  # (batch_size x seq_length x E)
        hidden = self.embedding_to_hidden(embeddings)    # (batch_size x seq_length x H)
        return hidden
