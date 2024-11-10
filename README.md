# BertJapaneseTokenizer介绍及对比（基于MindNLP实现）

## BERT简要介绍

BERT模型原作者在原文中指出，将预训练语言表示应用于下游任务有两种现有策略：基于特征和微调，而这两种方法一般都是采用的单向语言模型，每个token都只能关注Transformer 的自注意层中的先前token，因此很难完成一些需要理解上下文的任务（如问答任务）。

于是，作者提出了BERT来改进微调的方法，BERT全称*Bidirectional Encoder Representations from Transformers*，即一个多层双向的Transformer 编码器。它通过使用“掩码语言模型”（MLM） 预训练目标来缓解前面提到的问题。MLM随机掩码输入中的一些token，然后基于上下文来预测掩码，从而能够预训练双向transformer。

## BertJapaneseTokenizer介绍

一般来说，将文本划分为单词时，对于英语可以使用空格来划分单词。然而，在日语中，单词是连续书写的，因此不可能通过简单的处理来划分单词。

因此，BertJapaneseTokenizer使用Mecab进行形态分析，将其划分为单词，然后用WordPiece或Character方法将单词划分为子词，再将其转换为token id。

分词的相关配置可以在 模型目录下的*config.json* 和 *tokenizer_config.json* 中找到，下面来进行介绍。

### tokenizer_config.json

文件内容如下：

```json
{"do_lower_case": false, "word_tokenizer_type": "mecab", "subword_tokenizer_type": "character", "model_max_length": 512}
```

其中，*do_lower_case*表示设置是否将所有输入文本转换为小写。*word_tokenizer_type*中指定词级别的分词器为 `mecab`，这是一个开源的日本语分词工具，特别适用于日语文本。*subword_tokenizer_type*中设置子词级别的分词器为字符级分词器，即将单词进一步拆分成字符。最后*model_max_length*指定模型输入的最大长度为 512 。

### config.json

文件内容如下：

```json
{
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "tokenizer_class": "BertJapaneseTokenizer",
  "type_vocab_size": 2,
  "vocab_size": 32000
}
```

其中，涉及分词的主要是以下几个参数：

*tokenizer_class*中指定了分词器类为 `BertJapaneseTokenizer`，*vocab_size*中指定了词汇表的大小，即分词器支持的最大词汇单元数量为32000，*pad_token_id*指定了输入不足模型最大长度的部分补0。

### Mecab分词器

 Mecab 是一款适用于日语的形态分析软件。 下面是使用 Mecab的示例。

```py
#需要先从官网下载Mecab.exe安装，注意安装时勾选编码格式为UTF-8，否则最后会输出乱码
#然后再pip install mecab-python3即可
import MeCab

# 初始化 MeCab 分词器
mecab_tagger = MeCab.Tagger()

# 定义需要分词的文本
text = "明日は友達と映画を見に行きます。"

# 使用 MeCab 的 parse 函数进行分词
parsed_text = mecab_tagger.parse(text)

# 将分词结果去掉 EOS 行，得到实际的分词列表
tokens = parsed_text.splitlines()[:-1]

# 打印每个分词单元
for token in tokens:
    print(token)
```

输出如下，包含词性、原形等详细信息，可以根据需要处理这些分词结果。

```bash
明日    名詞,副詞可能,*,*,*,*,明日,アシタ,アシタ
は      助詞,係助詞,*,*,*,*,は,ハ,ワ
友達    名詞,一般,*,*,*,*,友達,トモダチ,トモダチ
と      助詞,並立助詞,*,*,*,*,と,ト,ト
映画    名詞,一般,*,*,*,*,映画,エイガ,エイガ
を      助詞,格助詞,一般,*,*,*,を,ヲ,ヲ
見      動詞,自立,*,*,一段,連用形,見る,ミ,ミ
に      助詞,格助詞,一般,*,*,*,に,ニ,ニ
行き    動詞,自立,*,*,五段・カ行促音便,連用形,行く,イキ,イキ
ます    助動詞,*,*,*,特殊・マス,基本形,ます,マス,マス
。      記号,句点,*,*,*,*,。,。,。
```

### 子词分词器

若直接将Mecab 划分的单词映射为token id，那么对应数目会非常多（显然词的数量是多于字的数量的）。因此，还需要进一步将单词划分为子词，作者给出了WordPiece和Character两种子词划分方法。bert_base_japanese中选用的是Character方法，该方法直接将每个词再按字符进行拆分，输出如下：

```
['明', '日']
['は']
['友', '達']
['と']
['映', '画']
['を']
['見']
['に']
['行', 'き']
['ま', 'す']
['。']
```

### 实际样例

```py
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
```

结果如下:

```bash
['明日', 'は', '友達', 'と', '映画', 'を', '見', 'に', '行き', 'ます', '。']  
[11475, 9, 12455, 13, 450, 11, 212, 7, 2609, 2610, 8]
```

## 与其他几种语言分词器的对比

### bert-base-chinese

在模型文件中的tokenizer.json中可以看到每个汉字对应的token id。

```py
from mindnlp.transformers import  BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    './model/bert-base-chinese'
)
text = "明天我要和朋友去看电影。"
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

print(tokenized_text)
print(indexed_tokens)
```

结果如下，可以看到是按字做的切分：

```bash
['明', '天', '我', '要', '和', '朋', '友', '去', '看', '电', '影', '。']
[3209, 1921, 2769, 6206, 1469, 3301, 1351, 1343, 4692, 4510, 2512, 511]
```

### bert-base-uncased

在模型文件中的tokenizer.json中可以看到每个单词对应的token id。

```python
from mindnlp.transformers import  BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    './model/bert-base-uncased'
)
text = "Tomorrow I'm going to watch a movie with my friends."
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

print(tokenized_text)
print(indexed_tokens)
```

结果如下，可以看到是单词做的切分：

```bash
['tomorrow', 'i', "'", 'm', 'going', 'to', 'watch', 'a', 'movie', 'with', 'my', 'friends', '.']
[4826, 1045, 1005, 1049, 2183, 2000, 3422, 1037, 3185, 2007, 2026, 2814, 1012]
```

## 参考文献

1. [BertJapaneseTokenizer ： 用于日本 BERT 的分词器 | 作者 Kazuki Kyakuno | AXINC | 中等](https://medium.com/axinc/bertjapanesetokenizer-日本語bert向けトークナイザ-7b54120aa245)

2. Devlin J. Bert: Pre-training of deep bidirectional transformers for language understanding[J]. arXiv preprint arXiv:1810.04805, 2018.