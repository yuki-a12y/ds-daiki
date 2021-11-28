import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from pandas.core.frame import DataFrame
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve, precision_recall_curve
from typing import Any
import pprint
from pathlib import Path

p = Path()
BASE_DIR = p.cwd().parent.parent


class ModelLgb:
    def __init__(self, run_fold_name, prms=None):
        self.booster = None
        self.run_fold_name = run_fold_name
        if prms is not None:
            self.params = prms
        else:
            self.params = {
                # 2値分類
                'objective': 'binary',
                # 乱数シードを指定
                'seed': 44,
                # 出力の間隔
                'verbose': 10,
                # カラムベースのヒストグラム
                'force_col_wise': True,
                # aucの最大化を目指す
                'metric': 'auc',
                'scale_pos_weight': 4.0
            }
        self.num_round = 1000
        self.early_stopping_rounds: int = 10

    def fit(self, tr_x, tr_y, va_x, va_y, categorical_feature):
        self.target_columns = tr_x.columns
        self.eval_results = {}
        print(self.target_columns)
        # データセットを変換
        lgb_train = lgb.Dataset(data=tr_x,
                                label=tr_y,
                                categorical_feature=categorical_feature)
        lgb_eval = lgb.Dataset(data=va_x,
                               label=va_y,
                               reference=lgb_train,
                               categorical_feature=categorical_feature)
        self.booster = lgb.train(params=self.params,
                                 train_set=lgb_train,
                                 categorical_feature=categorical_feature,
                                 num_boost_round=self.num_round,
                                 early_stopping_rounds=self.early_stopping_rounds,
                                 valid_names=['train', 'valid'],
                                 valid_sets=[lgb_train, lgb_eval],
                                 evals_result=self.eval_results,
                                 verbose_eval=2)

        return self.booster

    def predict(self, x):
        data = lgb.Dataset(x)
        y_pred = self.booster.predict(x, num_iteration=self.booster.best_iteration)
        return y_pred

    def get_feature_importance(self, target_columns=None):
        '''
        特徴量の出力
        '''
        if target_columns is not None:
            self.target_columns = target_columns
        feature_imp = pd.DataFrame(self.booster.feature_importance(),
                                   index=self.target_columns, columns=['fi'])

        return feature_imp

    def save_model(self):
        self.booster.save_model(filename=str(
            BASE_DIR.joinpath('model', f'{self.run_fold_name}.txt')))

    def load_model(self):
        self.booster = lgb.Booster(model_file=BASE_DIR.joinpath(
            'model', f'{self.run_fold_name}.txt'))

    def get_tree(self, ax):
        '''
        ベストイテレーションのツリー
        '''
        lgb.plot_tree(self.booster, ax=ax, tree_index=self.booster.best_iteration - 1)

    def get_metric(self, ax):
        '''
        metricのグラフ出力
        '''
        lgb.plot_metric(self.eval_results, ax=ax)


class Runner():
    def __init__(self, run_name: str, model_cls, data_train: DataFrame, categorical_feature: list, prms: dict = None, target_feature: str = None) -> None:
        '''
        モデルの学習クラス

        # 引数
        model: 自作のModelクラスを指定
        data_train: 目的変数と説明変数両方を含む学習データを指定
        categorical_feature: カテゴリ変数名を指定
        target_feature: 目的変数名を指定。指定しない場合は、デフォルトでclaimが指定される
        '''
        self.run_name = run_name
        self.model_cls = model_cls
        self.data_train = data_train
        self.prms = prms
        self.categorical_feature = categorical_feature
        if target_feature == None:
            self.target_feature = 'claim'
        else:
            self.target_feature = target_feature

    def run_train_fold(i_fold):
        pass

    def run_train_cv(self, k: int):
        '''
        クロスバリデーションでモデルの学習、予測を行う。

        ## 引数

        k: 分割数
        '''

        # 目的変数と説明変数に分ける
        train_x, train_y = self.data_train.drop(
            self.target_feature, axis=1), self.data_train[self.target_feature]

        # foldはループの何回目かを表す
        fold = 0

        # 予測値の入れ物（DataFrame）
        self.pred = pd.DataFrame(train_y.rename('label'))

        # 特徴量重要度の入れ物
        self.feature_imp = pd.DataFrame()

        # figure, ax作成
        fig_tree, axes_tree = plt.subplots((k + 1) // 2, 2, figsize=(15, 5), dpi=200, facecolor='w')
        fig_metric, axes_metric = plt.subplots(
            (k + 1) // 2, 2, figsize=(10, 10), dpi=200, facecolor='w')

        # ↓クラスの割合をそのままにランダムでk分割する
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=72)

        # クロスバリデーションのループ
        for tr_idx, va_idx in kf.split(train_x, train_y):

            # 何fold目か1足す
            fold += 1

            # トレーニング用とバリデーション用に分ける
            tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
            tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

            model = self.model_cls(run_fold_name=f'{self.run_name}_{fold}', prms=self.prms)

            # 学習
            model.fit(tr_x, tr_y, va_x, va_y, categorical_feature=self.categorical_feature)

            # バリデーションデータを予測
            va_pred = model.predict(va_x)

            # predに予測結果を入力
            s_pred = pd.Series(va_pred, index=va_y.index, name=f'pred_{fold}')
            self.pred = pd.concat([self.pred, s_pred], axis=1)

            # df_feature_impに特徴量重要度をmerge
            self.feature_imp = pd.merge(self.feature_imp, model.get_feature_importance().rename(
                columns={'fi': f'{fold}'}), how='outer', right_index=True, left_index=True)

            # treeの可視化
            axes_tree[(fold - 1) // 2, (fold - 1) % 2].set_title(f'{fold}')
            model.get_tree(axes_tree[(fold - 1) // 2, (fold - 1) % 2])

            # metricの可視化
            axes_metric[(fold - 1) // 2, (fold - 1) % 2].set_title(f'{fold}')
            model.get_metric(axes_metric[(fold - 1) // 2, (fold - 1) % 2])

            # モデルの保存
            model.save_model()

        # すべての結果をtotalにまとめる
        self.pred['total'] = self.pred.loc[:, 'pred_1':f'pred_{k}'].max(axis=1)

        # 平均を算出
        self.feature_imp['mean'] = self.feature_imp.mean(axis=1)

    def run_predict_cv(self, x_test: DataFrame) -> DataFrame:
        # 予測値の入れ物（DataFrame）
        df_pred = pd.DataFrame(index=x_test.index)

        # 各モデルのfoldごとの予測値を算出
        for fold in range(1, 6):
            # Modelクラスのインスタンスを作成
            model = self.model_cls(f'{self.run_name}_{fold}', prms=self.prms)
            # モデル読み込み
            model.load_model()
            # 予測値をDataFrameに入れる
            df_pred[f'{fold}'] = model.predict(x_test)

        return df_pred

    def get_pred(self, threshold):
        '''
        予測値を出力します
        thresholdは、TP、FP、TN、FNを判断するしきい値です。
        '''
        def cm(x, threshold):
            if x['label'] == 0:
                if x['total'] >= threshold:
                    return 'FP'
                if x['total'] < threshold:
                    return 'TN'
            else:
                if x['total'] >= threshold:
                    return 'TP'
                if x['total'] < threshold:
                    return 'FN'
        self.pred['cm'] = self.pred.apply(cm, threshold=threshold, axis=1)
        return self.pred

    def evaluating_all(self, threshold: float):
        '''
        まとめて表示
        各評価指標を算出
        混合行列、ROC曲線、PR曲線をまとめて表示

        ## 引数

        df_label_pred: 関数model_learningの戻り値pred
        threshold: 閾値
        '''

        # 混合行列
        self.plotting_confusion_matrix(threshold)

        # ROC
        self.plotting_roc()

        # PR
        self.plotting_pr()

        # classification report
        self.get_score(threshold)

    def get_feature_imp(self):
        return self.feature_imp

    def get_score(self, threshold: float):
        pprint.pprint(classification_report(self.pred['label'].values, self.pred['total'].map(
            lambda x: 1 if x >= threshold else 0), output_dict=True))

    def plotting_confusion_matrix(self, threshold: float):
        cm = confusion_matrix(self.pred['label'].values, self.pred['total'].map(
            lambda x: 1 if x >= threshold else 0))
        cm_matrix = pd.DataFrame(data=cm, columns=['Predict Negative:0', 'Predict Positive:1'], index=[
                                 'Actual Negative:0', 'Actual Positive:1'])
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
        plt.show()

    def plotting_roc(self):
        fpr, tpr, thresholds = roc_curve(self.pred['label'].values, self.pred['total'])
        score = auc(fpr, tpr)

        plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' % score)
        plt.plot(np.linspace(1, 0, len(fpr)), np.linspace(1, 0, len(fpr)),
                 label='Random ROC curve (area = %.2f)' % 0.5, linestyle='--', color='gray')

        plt.legend()
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        plt.show()

    def plotting_pr(self):
        precision, recall, thresholds = precision_recall_curve(
            self.pred['label'].values, self.pred['total'])

        score = auc(recall, precision)

        plt.plot(recall, precision, label='PR curve (area = %.2f)' % score)
        plt.legend()
        plt.title('PR curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)
        plt.show()
