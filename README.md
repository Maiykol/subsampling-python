# DAZER (DAtaset siZe Effect estimatoR)

## Class Subsampler
The 'Subsampler' class serves to subsample proportions of the data. While doing so, it is able to preserve the distribution of values in selected features (columns_keep_ratio). <br />
Additionally, it offers the functionality to extract a test dataset. Samples in this dataset will be excluded from following subsamples.

### setup & generate test data

```python
import dazer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
            n_samples=100, n_features=10, random_state=444)
df = pd.DataFrame(X)
df.join(pd.Series(y, name='label'))
```

### subsample

```python
subsampler = dazer.Subsampler(df, columns_keep_ratio=['label'], allowed_deviation=.2)

df_test = subsampler.extract_test()

df_test = subsampler.extract_test(test_size=.2, random_state=101)

df_train_1 = subsampler.subsample(subsample_factor=.1, random_state=101)
df_train_2 = subsampler.subsample(subsample_factor=.2, random_state=101)
df_train_3 = subsampler.subsample(subsample_factor=.3, random_state=101)
```

## Class Classifier

### prepare data for training and testing

```python
y_test = df_test[target_column] == target_value
X_test = df_test.drop([target_column], axis=1)

y_train = df_train_1[target_column] == target_value
X_train = df_train_1.drop([target_column], axis=1)
```

### model training and evaluation (Random Forest)

```python
classifier = dazer.Classifier(X_train, y_train, X_test, y_test)
model, evaluation = classifier.train_test('rf', scoring='f1')
```

### model training and evaluation (Multi-layer Perceptron)

```python
classifier = dazer.Classifier(X_train, y_train, X_test, y_test)
model, evaluation = classifier.train_test('mlp', scoring='f1', param_model={'solver': 'lbfgs', 'hidden_layer_sizes': (10, 5), 'random_state': 101, 'alpha': 1e-5, 'C': 1})
```

### model training and evaluation (Support Vector Classification with rbf kernel)

```python
classifier = dazer.Classifier(X_train, y_train, X_test, y_test)
model, evaluation = classifier.train_test('svc', scoring='f1', param_model={'kernel': 'rbf', 'C': 1, 'gamma': 2, 'random_state': 101})
```

### available models:
- 'rf' (<a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html" target="_blank">Random Forest</a>)
- 'xgb' (<a href="https://xgboost.readthedocs.io/en/stable/" target="_blank">XGBoost</a>)
- 'mlp' (<a href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html" target="_blank">Multi-layer Perceptron</a>)
- 'gnb' (<a href="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html" target="_blank">Gaussian Naive Bayes</a>)
- 'svc' (<a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html" target="_blank">Support Vector Classification</a>)


### save model immediately as .joblib object

```python
classifier = dazer.Classifier(X_train, y_train, X_test, y_test)
model, evaluation = classifier.train_test('rf', model_path='models/model_1.joblib', scoring='f1')
```


## Utils

Useful high level wrappers incorporating the dazer functionalities.

```python
test_dict, train_dict = dazer.subsample_iterative(df, columns_keep_ratio=[], allowed_deviation=.2, test_size=.2, random_states=[101, 102, 103, 104, 105], attempts=10000, ratios=[.2, .4, .6, .8, 1]):
```

## Run unittests

`python3 -m unittest discover tests`