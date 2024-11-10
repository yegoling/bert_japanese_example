from mindnlp.transformers import  BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    './model/bert-base-chinese'
)
text = "明天我要和朋友去看电影。"
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

print(tokenized_text)
print(indexed_tokens)