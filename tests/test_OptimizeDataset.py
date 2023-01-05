import pandas as pd
import pytest
from src.OptimizeDataset import OptimizeDataset as od


@pytest.fixture
def observation_df():
    return pd.DataFrame(
        data={
            "to_uint8": [1, 2],
            "to_uint16": [32000, 15000],
            "to_uint32": [64000, 100000],
            "to_float16": [0.1, 0.2],
            "to_float32": [32000.0, 15000.0],
            "to_float64": [64000000.012, 1000000000.011],
            "to_categorical": ["1", "1"],
            "to_non_categorical": ["0", "1"],
        }
    )


class TestOptimize:
    def test_optimize_integer_features(self, observation_df):
        result_df = od._optimize_integer_features(observation_df)
        result_dtype = result_df.dtypes

        # Expected
        expected_result = ["uint8", "uint16", "uint32"]
        assert result_dtype.to_list() == expected_result

    def test_optimize_float_features(self, observation_df):
        result_df = od._optimize_float_features(observation_df)
        result_dtype = result_df.dtypes

        # Expected
        expected_result = ["float32", "float32", "float64"]
        assert result_dtype.to_list() == expected_result

    def test__optimize_object_features(self, observation_df):
        result_df = od._optipmize_object_features(observation_df)
        result_dtype = result_df.dtypes

        # Expected
        expected_result = ["category", "object"]
        assert result_dtype.to_list() == expected_result
