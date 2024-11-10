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