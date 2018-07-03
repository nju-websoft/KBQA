# encoding=utf-8
import jieba
import jieba.posseg as pseg

class Word(object):
    """
    定义Word类结构，用于refo模板匹配
    """
    def __init__(self, token, pos):
        self.token = token
        self.pos = pos


class Tagger:
    """
    定义Tagger类，实现自然语言转为Word对象的方法。
    """
    def __init__(self, dict_paths):
        # 加载外部词典
        for p in dict_paths:
            jieba.load_userdict(p)

        # jieba不能正确切分的词语，我们人工调整其频率。
        jieba.suggest_freq(('喜剧', '电影'), True)
        jieba.suggest_freq(('恐怖', '电影'), True)
        jieba.suggest_freq(('科幻', '电影'), True)
        jieba.suggest_freq(('喜剧', '演员'), True)
        jieba.suggest_freq(('出生', '日期'), True)
        jieba.suggest_freq(('英文', '名字'), True)

    @staticmethod
    def get_word_objects(sentence):
        """
        把自然语言转为Word对象
        :param sentence:
        :return:
        """
        return [Word(word, tag) for word, tag in pseg.cut(sentence)]

if __name__ == '__main__':
    tagger = Tagger(['./external_dict/movie_title.txt', './external_dict/person_name.txt'])
    while True:
        s = input()
        for i in tagger.get_word_objects(s):
            print (i.token, i.pos)
