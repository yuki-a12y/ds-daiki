# Imbalanced Data-Set

## References

[Imbalanced Classes In Machine Learning](https://towardsdatascience.com/dealing-with-imbalanced-classes-in-machine-learning-d43d6fa19d2)[(->Japanese Translation)](https://qiita.com/r-takahama/items/631a59953fc20ceaf5d9)
[【ML Tech RPT. 】第4回 不均衡データ学習 (Learning from Imbalanced Data) を学ぶ(1)](https://buildersbox.corp-sansan.com/entry/2019/03/05/110000)
[不均衡データの扱い方と評価指標！SmoteをPythonで実装して検証していく！](https://toukei-lab.com/imbalance-data-smote)

## what is "Imbalanced Data-Set" ?

Not equal portion of data-set
For example, suppose I have two classes -- A and B. Class A is 90% of my data-set and class B is 10% of my data-set. In real-world, there are diseases markers in medical data and so on.
For such data-set, I can reach an accuracy of 90% by simply predicting class A every time. But, in many case, the goal is to identify the minority class.

## confusion matrix

![confusion matrix](/confusion-matrix.png)

```math
accuracy=\frac{TP+TN}{TP+TN+FP+FN}\\
error rate=1-accuracy\\
precision=\frac{TP}{TP+FP}\\
recall=\frac{TP}{TP+FN}\\
F1-score\\
Fβ-score\\
MCC\\

```

## Metrics

Generally, this problem deals with the trade-off between recall and precision. In situations where we want to detect instaces of a minority class, we are usually conserned more so with recall than precision. It may be that switching the metric you optimize for during parameter selection or model selection is enough to provide desirable performance detecting the minority class.

## Cost-sensitive Learning

In regular learning, we treat all misclassifications equally, which causes issues in imbalanced classification problems, as there is no extra reward for identifying the minority class over the majority class.
Cost-sensitive learning changes this, and confusion matrix. This allows us to penalize misclassifications of the minority class more heavily than we do with misclassifications of the majority class, in hopes that this increases the true positive rate. A common scheme for this is to have the cost equal to the inverse of the proportion of the data-set that the class makes up. This increases the penalization as the class size decreases.

## Sampling

A simple way to fix imbalanced data-sets is simply to balance them, either by oversampling instances of the minority class or undersampling instances of the majority class. However, oversampling the minority can lead to model overfitting, since it will introduce duplicate instances, drawing from a pool of instances that is already small. Similarly, undersampling the majority can end up leaving out important instances that provide important differences between the two classes.

There also exist more powerful sampling methods that go beyond simple oversampling or undersampling. The most well known example of this is SMOTE, which actually creates new instances of the minority class by forming convex combinations of neighboring instances. This allows us to balance our data-set without as much overfitting, as we create new synthetic examples rather than using duplicates. This however does not prevent all overfitting, as these are still created from existing data points.

## Anomaly Detection

In more extreme cases, it may be better to think of classification under the context of anomaly detection. In anomaly detection, we assume that there is a “normal” distribution(s) of data-points, and anything that sufficiently deviates from that distribution(s) is an anomaly. When we reframe our classification problem into an anomaly detection problem, we treat the majority class as the “normal” distribution of points, and the minority as anomalies. There are many algorithms for anomaly detection such as clustering methods, One-class SVMs, and Isolation Forests.
