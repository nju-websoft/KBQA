# encoding=utf-8
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import OrderedDict

class Virtuoso:
    """
        利用SOARQKWrapper向Virtuoso发送SPARQL查询，解析返回的结果
    """
    def __init__(self, endpoint_url='http://localhost:8890/sparql'):
        """
        初始化Virtuoso提供的SPARQL查询的url,默认为http://localhost:8890/sparql
        :param endpoint_url:
        """
        self.sparql_conn = SPARQLWrapper(endpoint_url)

    def get_sparql_result(self, query):
        """
        :param query: SPARQL 查询语句
        :return:
        """
        self.sparql_conn.setQuery(query)  # 设置查询语句
        self.sparql_conn.setReturnFormat(JSON)  # 设置返回结果格式
        return self.sparql_conn.query().convert()  # 返回结果，并转化为JSON格式

    def get_sparql_result_value(self, query_result):
        """
        解析查询结果，用列表存储结果的值
        :param query_result:
        :return:
        """
        try:
            query_head = query_result['head']['vars']  # 内含SPARQL中请求的变量
            query_results = list()
            for r in query_result['results']['bindings']:  # 内含返回的结果
                temp_dict = OrderedDict()  # key是请求的变量，value是该变量查询的结果
                for h in query_head:
                    temp_dict[h] = r[h]['value']
                query_results.append(temp_dict)
            values = list()  # 将所有结果合并
            for qr in query_results:
                for _, value in qr.items():
                    values.append(value)
            return values  # 返回结果
        except KeyError:  # 是ASK类查询， {'head': {'link': []}, 'boolean': True}
            return query_result['boolean']  # ASK类查询

if __name__ == '__main__':
    virtuo = Virtuoso()
    my_query = """
        PREFIX : <http://www.kgdemo.com#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        ASK {
            ?s :personName '周星驰'.
            ?s :hasActedIn ?m.
            ?m :movieTitle ?x
        }
    """
    result = virtuo.get_sparql_result(my_query)
    print(result)