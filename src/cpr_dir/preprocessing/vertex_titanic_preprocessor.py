from pandas import DataFrame, get_dummies
from typing import Union
from copy import deepcopy


class TitanicPreprocessor(object):
    DATA_CONFIG = {'features': {'fixed': ['Pclass', 'Age', 'SibSp', 'Parch'],
                                'age_col': 'Age',
                                'gender_col': 'Sex'
                                },
                   'target_col': 'Survived',
                   'passenger_id': 'PassengerId'
                   }

    def __init__(self):
        return

    def preprocess(self,
                   df: DataFrame,
                   # age_col: str,
                   # gender_col: str,
                   # fixed_columns: list
                   ) -> DataFrame:
        # a small analysis showed that the avg age is around 29.6
        features = self.DATA_CONFIG['features']
        age_col = features['age_col']
        gender_col = features['gender_col']
        fixed_columns = features['fixed']

        dg = deepcopy(df)
        dg = self.fill_na(df=dg, default_value=29.6, col=age_col)
        dg = self.women_children_first_rule(df=dg,
                                            age_col=age_col,
                                            gender_col=gender_col,
                                            female_value='female',
                                            new_col_name='women_children_first_rule_eligible',
                                            scale=5)
        gender_values = list(df[gender_col].sort_values(ascending=True).unique())
        dg = self.dummify_categorical(df=dg, col=gender_col, scale=4)

        return dg[fixed_columns + ['women_children_first_rule_eligible'] + gender_values[:-1]]

    @staticmethod
    def dummify_categorical(df: DataFrame,
                            col: str,
                            prefix: str = '',
                            prefix_sep: str = '',
                            scale: int = 1
                            ) -> DataFrame:
        values = df[col].unique()
        df_dummify = get_dummies(df, prefix=prefix, prefix_sep=prefix_sep, columns=[col])
        new_cols = [f'{prefix}{prefix_sep}{val}' for val in values]
        df_dummify[new_cols] = df_dummify[new_cols].mul(scale)
        return df_dummify

    @staticmethod
    def fill_na(df: DataFrame,
                col: str,
                default_value: Union[str, int, float]
                ) -> DataFrame:
        dg = deepcopy(df)
        dg[col].fillna(value=default_value, inplace=True)
        return dg

    @staticmethod
    def women_children_first_rule(df: DataFrame,
                                  age_col: str,
                                  gender_col: str,
                                  female_value: Union[str, int],
                                  new_col_name: str,
                                  scale: int = 1
                                  ) -> DataFrame:
        dg = deepcopy(df)
        dg[new_col_name] = dg[[age_col, gender_col]].apply(lambda r: scale*int(r[0] < 18 or r[1] == female_value),
                                                           axis=1)
        return dg

