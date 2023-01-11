import pandas as pd
import pytest
import src.memoframe as mf


@pytest.fixture
def observation_df():
    return pd.DataFrame(
        data={
            "to_uint8": [1, 2, 3],
            "to_uint16": [32000, 15000, 16000],
            "to_uint32": [64000, 100000, 68000],
            "to_float16": [0.1, 0.2, 0.12],
            "to_float32": [32000.0, 15000.0, 16000.0],
            "to_float64": [
                64000000.012,
                1000000000.011,
                111111111111111111111.111111111,
            ],
            "to_categorical": ["1", "1", "1"],
            "to_non_categorical": ["0", "1", "2"],
        }
    )


class TestMemoFrame:
    def test_optimize_integer_features(self, observation_df):
        result_df = mf._optimize_integer_features(observation_df)
        result_dtype = result_df.dtypes

        # Expected
        expected_result = ["uint8", "uint16", "uint32"]
        assert result_dtype.to_list() == expected_result

    def test_optimize_float_features(self, observation_df):
        result_df = mf._optimize_float_features(observation_df)
        result_dtype = result_df.dtypes

        # Expected
        expected_result = ["float32", "float32", "float64"]
        assert result_dtype.to_list() == expected_result

    def test_optimize_object_features(self, observation_df):
        result_df = mf._optimize_object_features(observation_df)
        result_dtype = result_df.dtypes
        print(result_dtype)

        # Expected
        expected_result = ["category", "object"]
        assert result_dtype.to_list() == expected_result

    def test_downsize_memory(self, observation_df):
        result_df = mf.downsize_memory(observation_df)
        result_dtype = result_df.dtypes

        # Expected
        expected_result = [
            "category",
            "object",
            "float32",
            "float32",
            "float64",
            "uint8",
            "uint16",
            "uint32",
        ]
        assert result_dtype.to_list() == expected_result

    def test_get_opti_info(self, observation_df):
        result_str = mf.get_opti_info(observation_df)

        # Expected
        expected_result = "Up to: 13 % memory usage can be saved"
        assert expected_result == result_str
