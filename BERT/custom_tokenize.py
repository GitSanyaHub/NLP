from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def align_labels_and_tokens(word_ids, labels):
    """ Aligns tokens and their respective labels

    Args:
        word_ids (list): word ids of tokens after subword tokenization.
        labels (list): original labels irrespective of subword tokenization.

    Returns:
        updated_labels (list): labels aligned with respective tokens.

    """

    updated_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            updated_labels.append(-100 if word_id is None else labels[word_id])
        elif word_id is None:
            updated_labels.append(-100)
        else:
            label = labels[word_id]
            # B-XXX to I-XXX for subwords (Inner entities)
            if label % 2 == 1:
                label += 1
            updated_labels.append(label)
    return updated_labels


def tokenize_dataset(dataset):
    
    """ Performs tokenization and aligns all tokens and labels
        in the dataset.
    
    Args:
        dataset (DatasetDict): dataset containing tokens and labels.
    
    Returns:
        tokenized_data (dict): contains input_ids, attention_mask, token_type_ids, labels
        
    """
    
    tokenized_data = tokenizer(dataset["tokens"], truncation=True, is_split_into_words=True)
    all_labels = dataset["ner_tags"]
    updated_labels = []
    for i, labels in enumerate(all_labels):
        updated_labels.append(align_labels_and_tokens(tokenized_data.word_ids(i), labels))
    tokenized_data["labels"] = updated_labels
    return tokenized_data
