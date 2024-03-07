import pandas as pd
from sklearn.metrics import  classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV

data=pd.read_csv('preprocessed.csv',index_col=0)
X=data.drop(['aki'],axis=1)
y=data['aki']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
smote_sampler = SMOTE(random_state=12,sampling_strategy='minority')
X_train, y_train= smote_sampler.fit_resample(X_train, y_train)
#Decision Tree
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
stratifiedCV = StratifiedKFold(n_splits=5)
BestParams = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=stratifiedCV ,
    scoring='f1',
    n_jobs=-1
)
BestParams.fit(X_train, y_train)
best_dt = BestParams.best_estimator_
selector = SelectFromModel(estimator=best_dt, prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
selected_features = X_train.columns[selector.get_support()]
best_dt.fit(X_train_selected, y_train)
y_pred_test = best_dt.predict(X_test_selected)
report = classification_report(y_test, y_pred_test)
print(report)

#Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
model = SelectFromModel(rf, prefit=True)
X_train_selected = model.transform(X_train)
X_test_selected = model.transform(X_test)
selected_feature = X_train.columns[model.get_support()].tolist()
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
stratifiedCV = StratifiedKFold(n_splits=5)
BestParams = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=stratifiedCV ,
    scoring='f1',
    n_jobs=-1
)
BestParams.fit(X_train_selected, y_train)
best_rf =  BestParams.best_estimator_
y_pred_test = best_rf.predict(X_test_selected)
report = classification_report(y_test, y_pred_test)
print(report)

# AdaBoost
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)
model = SelectFromModel(ada, prefit=True)
X_train_selected = model.transform(X_train)
X_test_selected = model.transform(X_test)
selected_feature= X_train.columns[model.get_support()].tolist()
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1]
}
stratifiedCV = StratifiedKFold(n_splits=5)
BestParams = GridSearchCV(
    estimator=AdaBoostClassifier(random_state=42),
    param_grid=param_grid,
    cv=stratifiedCV ,
    scoring='f1',
    n_jobs=-1
)
BestParams.fit(X_train_selected, y_train)
best_ada =  BestParams.best_estimator_
y_pred_test = best_ada.predict(X_test_selected)
report = classification_report(y_test, y_pred_test)
print(report)
# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
model = SelectFromModel(xgb, prefit=True)
X_train_selected = model.transform(X_train)
X_test_selected = model.transform(X_test)
selected_feature= X_train.columns[model.get_support()].tolist()
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
stratifiedCV = StratifiedKFold(n_splits=5)
BestParams = GridSearchCV(
    estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    param_grid=param_grid,
    cv=stratifiedCV ,
    scoring='f1',
    n_jobs=-1
)
BestParams.fit(X_train_selected, y_train)
best_xgb = BestParams.best_estimator_
y_pred_test = best_xgb.predict(X_test_selected)
report = classification_report(y_test, y_pred_test)
print(report)