import pandas as pd


class OptimizeDataset:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def _optimize_integer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe_int = dataframe.select_dtypes(include=["int"])
        converted_int = dataframe_int.apply(pd.to_numeric, downcast="unsigned")
        return converted_int

    def _optimize_float_features(dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe_float = dataframe.select_dtypes(include=["float"])
        converted_float = dataframe_float.apply(pd.to_numeric, downcast="float")
        return converted_float

    def _optipmize_object_features(dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe_obj = dataframe.select_dtypes(include=["object"])
        converted_obj = pd.DataFrame()
        for col in dataframe_obj.columns:
            num_unique_values = len(dataframe_obj[col].unique())
            num_total_values = len(dataframe_obj[col])
            if num_unique_values / num_total_values < 0.6:
                converted_obj.loc[:, col] = dataframe_obj[col].astype("category")
            else:
                converted_obj.loc[:, col] = dataframe_obj[col]
        return converted_obj

    @staticmethod
    def downsize_memory(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Pandas dataframe memory optimization
        df: pd.dataframe
        return: df_optimized : pd.dataframe
        """
        # optimize int features
        converted_int = OptimizeDataset._optimize_integer_features(dataframe)
        # optimize float features
        converted_float = OptimizeDataset._optimize_float_features(dataframe)
        # optimize object features
        converted_obj = OptimizeDataset._optipmize_object_features(dataframe)

        # concatanate optimized features
        df_optimized = pd.concat(
            [converted_obj, converted_float, converted_int], axis=1
        )

        return df_optimized
