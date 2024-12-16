class ChatHistory:
    def __init__(self):
        self.messages = []

    def add_user_message(self, content: str):
        # добавить новые сообщения пользователя
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        # добавить ответ системы
        self.messages.append({"role": "assistant", "content": content})

    def get_all_messages(self):
        # получить историю общения
        return self.messages

    def get_interactions_count(self):
        # возвращает количество полных взаимодействий: пользователь и ассистент
        # каждое взаимодействие состоит из двух сообщений.
        return len(self.messages) // 2

    def get_last_interactions(self, n: int):
        # получение последние n взаимодействий
        total_interactions = self.get_interactions_count()
        interactions_to_return = min(n, total_interactions)
        start_index = -2 * interactions_to_return
        return self.messages[start_index:] if interactions_to_return > 0 else []