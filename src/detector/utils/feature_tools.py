import copy
from pyspark.ml.feature import VectorAssembler



class FeatureTools():
    """Its methods allow to do preprocessing operation on dataset.  """

    @staticmethod
    def normalize_features(df_in, cols, sc):
        """

        :param df: pyspark.sql.dataframe.DataFrame
        :param cols: list of columns to be scaled
        :param sc: Standard Scaler (from pyspark.ml.feature, sklearn.preprocessing or similar)

        :return: df: Pandas.DataFrame, sc: trained scaler
        """
        #df = df_in.copy()
        assembler = VectorAssembler().setInputCols(df_in.columns).setOutputCol("features")
        transformed = assembler.transform(df_in)
        scaledModel = sc.fit(transformed.select("features"))
        df_scaled = scaledModel.transform(transformed) # TO DO: add possibility to scale only specific features.

        return df_scaled, scaledModel

    @staticmethod
    def other_():
        pass



    def fit(self, df_inp, target_col, numerical_columns, sc):
        """

        :param df_inp: pyspark.sql.dataframe.DataFrame
        :param target_col: str
        :param numerical_columns: list with the numerical columns
        :param sc: Scaler, from pyspark.ml.feature or similar

        """

        #df = df_inp.copy()
        self.numerical_columns = numerical_columns

        df, self.sc = self.normalize_features(df_inp, numerical_columns, sc)

        self.target = df.select(target_col).rdd
        self.data = df.drop(target_col).rdd
        self.colnames = df.columns

        return self

    def transform(self, df_inp, trained_sc=None):
        """

        :param df_inp: pyspark.sql.dataframe.DataFrame
        :param trained_sc: Scaler, from pyspark.ml.features or similar

        :return:  df: Pandas.DataFrame: transformed DataFrame
        """
        #df = df_inp.copy()
        if trained_sc:
            sc = copy.deepcopy(trained_sc)
        else:
            sc = copy.deepcopy(self.sc)

        df, _ = self.normalize_features(df_inp, self.numerical_columns, sc)

        return df



