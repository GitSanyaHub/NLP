def llm_answer(generation_pipeline, query, context, temperature=0.7, top_p=0.9, max_new_tokens=1024):
    recipes = "\n".join([
        f"- {recipe['dish_name']}\n  Ингредиенты: {''.join(recipe['ingredients'])}\n  Шаги: {', '.join(recipe['steps_for_cook'])}"
        for recipe in context
    ])

    prompt = f"""
Вопрос: {query}

Названия блюд, ингридиенты для приготовления и шаги готовки:
{recipes}
"""
    system_prompt = """Ты — шутливый и доброжелательный ассистент-повар. У тебя есть дополнительная информация о блюдах, ингредиентах и шагах приготовления, известная из контекста запроса. 

Твоя задача:
- Ответить, основываясь ТОЛЬКО на предоставленной информации. 
- Не придумывать детали, не указанные в данных.
- Ограничиться ОДНИМ рецептом из доступной информации.
- Соблюдать нормы русского языка.

Формат ответа:
1. Короткое приветствие (например: "Привет!").
2. Название блюда, о котором идёт речь.
3. Список ингредиентов блюда, разделённый запятыми или перечислением.
4. Подробные шаги приготовления, по порядку.
5. Заключение с добрым пожеланием (например: "Приятного аппетита!").

Пример:
Если в данных упомянут рецепт борща, сначала поздоровайся, потом назови блюдо "Борщ", перечисли ингредиенты, затем опиши шаги приготовления и закончи фразой "Приятного аппетита!".

Теперь, используя предоставленную информацию по запросу, составь ответ в указанном формате.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    output = generation_pipeline(messages, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p)

    return output[0]['generated_text'][-1]['content']



def predict(search_recipes_by_name, embedding_model,lsh,names, text_chunks_numbered , device, chat_history, generation_pipeline, query):
    print("Поиск рецептов...")
    context = search_recipes_by_name(
        query=query,
        embedding_model=embedding_model,
        lsh=lsh,
        names=names,
        text_chunks_numbered=text_chunks_numbered,
        device=device,
        max_results=3
    )

    print(f"Найдено рецептов: {len(context)}")
    print(f"Контекст: {context} \n")
    print(f"Думаю над рецептом... \n")

    answer = llm_answer(generation_pipeline, query, context)

    chat_history.add_user_message(query)
    chat_history.add_assistant_message(answer)

    return answer


def rewrite_query(query, chat_history, generation_pipeline, temperature=0.7, top_p=0.9, max_new_tokens=1024, n=1):
  last_interactions = chat_history.get_last_interactions(n)
  history_str = "\n".join([
    f"{msg['role']}: {msg['content']}" for msg in last_interactions
  ])

  prompt = f"""
Вопрос: {query}

История общения с пользователем:
{history_str}
"""
  system_prompt = """Ты ассистент, который должен переформулировать вопрос пользователя на основе предыдущей истории диалога.

- Прочитай историю общения с пользователем и исходный вопрос.
- Изучи контекст вопросов из истории.
- Переформулируй текущий вопрос так, чтобы он был максимально ясным и конкретным, отражая контекст из предыдущих сообщений.
- Твой ответ должен содержать ТОЛЬКО переформулированный вопрос и ничего больше.
- Не добавляй приветствия, комментарии, объяснения или другие слова.
- Не повторяй исходный вопрос.
- Не используй метки «ВОПРОС:» или другие дополнительные символы. Просто выведи итоговый переформулированный вопрос одной строкой.

Пример:  
История: Пользователь спрашивал о «Курином Супе».  
Исходный вопрос: «Какие ингредиенты нужны для его приготовления?».  
Желаемый ответ: «Какие ингредиенты нужны для приготовления Куриного Супа?»

Теперь прими мою историю и вопрос, и выведи только переформулированный вопрос в нужном формате.
"""

  messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": prompt},
  ]

  output = generation_pipeline(messages, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p)

  return output[0]['generated_text'][-1]['content']