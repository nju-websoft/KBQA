# encoding=utf-8
import question_temp
import word_tagging

class Question2Sparql:
    """
    将自然语言转为SPARQL查询语句
    """
    def __init__(self, dict_paths):
        self.tw = word_tagging.Tagger(dict_paths)
        self.rules = question_temp.rules  # 获取模板匹配规则

    def get_sparql(self, question):
        """
        进行语义解析，找到匹配的模板，返回对应的SPARQL查询语句
        :param question:
        :return:
        """
        word_objects = self.tw.get_word_objects(question)  # 将问题转换为word对象
        queries_dict = dict()

        for rule in self.rules:
            query, num = rule.apply(word_objects)  # 将规则应用到问句中，返回查询语句和条件数量

            if query is not None:
                queries_dict[num] = query

        if len(queries_dict) == 0:
            return None
        elif len(queries_dict) == 1:
            for a in queries_dict.values():
                return a
        else:
            # 匹配多个语句，以匹配关键词最多的句子作为返回结果
            sorted_dict = sorted(queries_dict.items(), key=lambda item: item[0], reverse=True)
            return sorted_dict[0][1]
