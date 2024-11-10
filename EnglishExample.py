from mindnlp.transformers import  BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    './model/bert-base-uncased'
)
text = "Tomorrow I'm going to watch a movie with my friends."
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

print(tokenized_text)
print(indexed_tokens)