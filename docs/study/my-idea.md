# My Idea

## Reference

[Kaggleで世界11位になったデータ解析手法～Sansan高際睦起の模範コードに学ぶ](https://eh-career.com/engineerhub/entry/2018/08/24/110000)
[不均衡データの扱い方と評価指標！SmoteをPythonで実装して検証していく](https://toukei-lab.com/imbalance-data-smote)

## Idea

1. EDA(Explanatory Data Analysis)
   [What is EDA ?](https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15)
2. feature extraction and Modeling

## Sampling

- Undersampling
- Oversampling
- Smote

## Select model

- GDBT
- Nural Network
- Linear

base model -> GBDT
+
ensemble NN or Linear

## Cost-sensitive Learning

[teratail](https://teratail.com/questions/114701)

sklearn -> sample_weight
XGBoost -> scale_pos_weight
LightGBM -> class_weight
Pytorch -> CrossEntropyLoss(weight)

## Text detection

## Anomaly Detection

[qiita](https://qiita.com/kyohashi/items/c3343de3cfa236df3bda)

## Treat as time series data
