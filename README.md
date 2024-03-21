# DAZER (DAtaset siZe Effect estimatoR)


## Example

```python
from dazer import Subsampler 
import seaborn as sns

target_column = 'y'

df = sns.load_dataset('penguins', data_home=self.test_dir)
df = df.dropna()
df[target_column] = df['species'] == 'Adelie'

subsampler = dazer.Subsampler(df, ['body_mass_g', 'y'], .07, True)

df_test = subsampler.extract_test(.2, random_state=2)
df_train = subsampler.subsample(.4, random_state=3)

y_test = df_test[target_column]
X_test = df_test.drop([target_column], axis=1)

y_train = df_train[target_column]
X_train = df_train.drop([target_column], axis=1)

classifier = dazer.Classifier(X_train, y_train, X_test, y_test)

model, evaluation = classifier.train_test('rf', scoring='f1')

print(evaluation)
```


## Class Subsampler
The 'Subsampler' class serves to subsample proportions of the data. While doing so, it is able to preserve the distribution of values in selected features (columns_keep_ratio). <br />
Additionally, it offers the functionality to extract a test dataset. Samples in this dataset will be excluded from following subsamples.

### setup & generate test data

```python
import dazer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd

X, y = make_classification(
            n_samples=100, n_features=10, random_state=444)
df = pd.DataFrame(X)
df = df.join(pd.Series(y, name='label'))
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

The class 'Classifier' contains wrappers for a number of classification models. Currently supported models are:
- 'rf' (<a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html" target="_blank">Random Forest</a>)
- 'xgb' (<a href="https://xgboost.readthedocs.io/en/stable/" target="_blank">XGBoost</a>)
- 'mlp' (<a href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html" target="_blank">Multi-layer Perceptron</a>)
- 'gnb' (<a href="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html" target="_blank">Gaussian Naive Bayes</a>)
- 'svc' (<a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html" target="_blank">Support Vector Classification</a>)

### prepare data for training and testing

```python
y_test = df_test[target_column] == target_value
X_test = df_test.drop([target_column], axis=1)

y_train = df_train_1[target_column] == target_value
X_train = df_train_1.drop([target_column], axis=1)
```

### model training and evaluation (example: Random Forest)

```python
classifier = dazer.Classifier(X_train, y_train, X_test, y_test)
model, evaluation = classifier.train_test('rf', scoring='f1')
```

For the possible scoring options, refer to https://scikit-learn.org/stable/modules/model_evaluation.html.

### model training and evaluation (Multi-layer Perceptron)

```python
classifier = dazer.Classifier(X_train, y_train, X_test, y_test)
model, evaluation = classifier.train_test('mlp', scoring='f1', param_grid={'solver': 'lbfgs', 'hidden_layer_sizes': (10, 5), 'random_state': 101, 'alpha': 1e-5, 'C': 1})
```

### model training and evaluation (Support Vector Classification with rbf kernel)

```python
classifier = dazer.Classifier(X_train, y_train, X_test, y_test)
model, evaluation = classifier.train_test('svc', scoring='f1', param_grid={'kernel': 'rbf', 'C': 1, 'gamma': 2, 'random_state': 101})
```


### save model immediately as .joblib object

```python
classifier = dazer.Classifier(X_train, y_train, X_test, y_test)
model, evaluation = classifier.train_test('rf', model_path='models/model_1.joblib', scoring='f1')
```

## Class Regressor

The regression models are similarly implemented to classification models of the class 'Classifier' above.
Currently supported models are:
- 'rf' (<a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html" target="_blank">Random Forest</a>)
- 'xgb' (<a href="https://xgboost.readthedocs.io/en/stable/" target="_blank">XGBoost</a>)
- 'mlp' (<a href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html" target="_blank">Multi-layer Perceptron</a>)
- 'svr' (<a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html" target="_blank">Support Vector Classification</a>)

### imports and generate test data
```python
import dazer
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd

X, y = make_regression(n_samples=1000, n_features=10, noise=1, random_state=444)
target_column = 'y'
df = pd.DataFrame(X)
df = df.join(pd.Series(y, name=target_column))
```

### prepare data for training and testing with dazer

```python
subsampler = dazer.Subsampler(df, columns_keep_ratio=[target_column], allowed_deviation=.2)

df_test = subsampler.extract_test(test_size=.2, random_state=101)
df_train = subsampler.subsample(subsample_factor=.1, random_state=101)

y_test = df_test[target_column]
X_test = df_test.drop([target_column], axis=1)

y_train = df_train[target_column]
X_train = df_train.drop([target_column], axis=1)
```

### model training and evaluation (example: Random Forest)

```python
regressor = dazer.Regressor(X_train, y_train, X_test, y_test)
model, evaluation = regressor.train_test('rf', scoring='max_error')
```

For the possible scoring options, refer to https://scikit-learn.org/stable/modules/model_evaluation.html.

### save model immediately as .joblib object

```python
regressor = dazer.Regressor(X_train, y_train, X_test, y_test)
model, evaluation = regressor.train_test('rf', model_path='models/model_1.joblib', scoring='max_error')
```



## Utils

Useful high level wrappers incorporating the dazer functionalities.

```python
test_dict, train_dict = dazer.subsample_iterative(df, columns_keep_ratio=[], allowed_deviation=.2, test_size=.2, random_states=[101, 102, 103, 104, 105], attempts=10000, ratios=[.2, .4, .6, .8, 1]):
```

## Run unittests

`python3 -m unittest discover tests`