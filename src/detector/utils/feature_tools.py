import copy




class FeatureTools():
    """Its methods allow to do preprocessing operation on dataset.  """

    @staticmethod
    def normalize_features(df_in, cols, sc):
        """

        :param df: Pandas.DataFrame
        :param cols: list of columns to be scaled
        :param sc: Standard Scaler (from pyspark.ml.feature, sklearn.preprocessing or similar)

        :return: df: Pandas.DataFrame, sc: trained scaler
        """
        #df = df_in.copy()
        scaledModel = sc.fit(df_in[cols])
        df_scaled = scaledModel.transform(df_in[cols])  # TO DO: add possibility to scale only specific features.
        return df_scaled, scaledModel

    @staticmethod
    def other_():
        pass



    def fit(self, df_inp, target_col, numerical_columns, sc):
        """

        :param df_inp: Pandas.DataFrame
        :param target_col: str
        :param numerical_columns: list with the numerical columns
        :param sc: Scaler, from pyspark.ml.feature or similar

        """

        #df = df_inp.copy()
        self.numerical_columns = numerical_columns

        df, self.sc = self.normalize_features(df_inp, numerical_columns, sc)

        self.target = df.drop(target_col, axis=1, inplace=True)
        self.data = df
        self.colnames = df.columns.to_list()

        return self

    def transform(self, df_inp, trained_sc=None):
        """

        :param df_inp: Pandas.DataFrame
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



