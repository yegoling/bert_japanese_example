#需要先pip install fugashi , pip install ipadic
#fugashi是 Mecab 的包装器,提供了对 MeCab 的接口。
#ipadic 是一个用于 MeCab 的日语词典，它提供了词汇、词性、原型等信息，供分词和分析使用。
from mindnlp.transformers import  BertJapaneseTokenizer

tokenizer = BertJapaneseTokenizer.from_pretrained(
    './model/bert-base-japanese'
)
text = "明日は友達と映画を見に行きます。"
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

print(tokenized_text)
print(indexed_tokens)