from typing import Any
from pandas.core.frame import DataFrame
from sklearn.preprocessing import LabelEncoder
import msoffcrypto  # https://github.com/nolze/msoffcrypto-tool
import io
import pandas as pd
import numpy as np


def read_excel_with_pass(file_path: str, password: str) -> DataFrame:
    '''
    パスワード付きのエクセルファイルを読み込む関数
    戻り値: pandas.DataFrame
    file_path: ファイルパス
    password: ファイルのパスワード
    '''
    decrypted = io.BytesIO()

    with open(file_path, "rb") as f:  # open()→https://docs.python.org/ja/3/library/functions.html#open
        file = msoffcrypto.OfficeFile(f)
        file.load_key(password=password)  # Use password
        file.decrypt(decrypted)

    df = pd.read_excel(decrypted, index_col=None)
    return df


def label_encording(df: DataFrame, cols: list) -> DataFrame:
    '''
    ラベルエンコーディングする関数

    ##　戻り値

    pandas.DataFrame

    ## 引数

    df: 対象のデータフレーム
    cols: ラベルエンコーディングする列のリスト
    '''

    for c in cols:
        le = LabelEncoder()
        le.fit(df[c])
        # ラベルを整数に変換
        df[c] = le.transform(df[c])

    return df


def frequency_encoding(df: DataFrame, cols: list, replace: bool = False) -> DataFrame:
    '''
    frequency encoding

    ## 引数

    df: データフレーム
    cols: frequency encodingするカラムのリスト
    replace: True→dfを置換する
    '''
    def core(df: DataFrame, cols: list) -> None:
        # カラムの数だけループ
        for c in cols:
            # 出現回数を求める
            freq = df[c].value_counts()
            # 置換する
            df[c] = df[c].map(freq)

    if replace == True:
        core(df, cols)
        return df
    else:
        df_copy = df.copy()
        core(df_copy, cols)
        return df_copy


def add_log_diff_BOD(df) -> None:
    '''
    BODの差の対数値
    diff_log_BODカラムを生成
    '''
    df['diff_log_BOD'] = (df['IN-BOD'] / df['OUT-BOD']).map(np.log10)
    return df
