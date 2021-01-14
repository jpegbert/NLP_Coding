from pyspark.ml.feature import Word2Vec, Word2VecModel

from common.ContentPartition import ContentPartition
from common.SparkSessionBase import SparkSessionBase


"""
原文链接：https://mp.weixin.qq.com/s/8_Xg34auT9WgZPdU-K9thQ
"""


class ContentVectorModel():
    content_vector_path = "hdfs://data1:8020/recommend/models/VECTOR_old.model"

    def __init__(self):
        self.spark = SparkSessionBase().create_spark_session()

    @staticmethod
    def get_model():
        word2vec_model = Word2VecModel.load(path=ContentVectorModel.content_vector_path)
        return word2vec_model

    # 获取训练数据集
    def _get_tran_data(self):
        basic_content = self.spark.sql(
            """
                SELECT cp.id publish_id,
                       cp.content_id,
                       cp.channel_id,
                       get_json_object(cc.content,'$.title') sentence
                from ods.content_publish cp,
                     ods.content_content cc
                WHERE cp.content_id = cc.id
                order by cp.id
            """
            )
        # spark读取文章内容并分词
        words_df = basic_content.rdd.mapPartitions(ContentPartition.segmentation).toDF(["publish_id", "content_id", "channel_id", "words"])
        return words_df

    def fit_model(self):
        train_data = self._get_tran_data()
        word2Vec = Word2Vec(vectorSize=1000, inputCol="words", outputCol="model", minCount=3, windowSize=5)
        word2Vec_model = word2Vec.fit(train_data)
        word2Vec_model.write().overwrite().save(ContentVectorModel.content_vector_path)
        return word2Vec_model


if __name__ == '__main__':
    csm = ContentVectorModel()
    csm.fit_model()

