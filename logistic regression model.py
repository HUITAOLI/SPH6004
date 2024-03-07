import pandas as pd
import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from genetic_selection import GeneticSelectionCV

data=pd.read_csv('preprocessed.csv',index_col=0)
X=data.drop(['aki'],axis=1)
y=data['aki']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
smote_sampler = SMOTE(random_state=12,sampling_strategy='minority')
X_train, y_train= smote_sampler.fit_resample(X_train, y_train)
#RFE+logistic regression model
log_reg = LogisticRegression(solver='liblinear')
rfe = RFE(estimator=log_reg, n_features_to_select=40)
rfe.fit(X_train, y_train)
selected_features = X_train.columns[rfe.support_]
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
log_reg.fit(X_train_selected, y_train)
y_pred = log_reg.predict(X_test_selected)
report = classification_report(y_test, y_pred)
print(report)
#L1 regularization+logistic regression model
log_reg_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.002,random_state=42, max_iter=1000)
log_reg_l1.fit(X_train, y_train)
non_zero_weights = np.sum(log_reg_l1.coef_ != 0, axis=1)
selected_features = X_train.columns[np.where(log_reg_l1.coef_.flatten() != 0)[0]].tolist()
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
log_reg.fit(X_train_selected, y_train)
y_pred = log_reg.predict(X_test_selected)
report = classification_report(y_test, y_pred)
print(report)
#Forward selection+logistic regression model
X_sample, X_unused, y_sample, y_unused = train_test_split(X, y, train_size=3000, stratify=y, random_state=42)
smote_sampler = SMOTE(random_state=12,sampling_strategy='minority')
X_sample,  y_sample= smote_sampler.fit_resample(X_sample,  y_sample)
log_reg = LogisticRegression(solver='liblinear')
sfs = SequentialFeatureSelector(log_reg, n_features_to_select="auto", direction='forward', cv=5)
sfs.fit(X_sample, y_sample)
selected_feature = X_sample.columns[sfs.get_support()].tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_train, y_train= smote_sampler.fit_resample(X_train, y_train)
X_train_selected = X_train[selected_feature]
X_test_selected = X_test[selected_feature]
log_reg.fit(X_train_selected, y_train)
y_pred = log_reg.predict(X_test_selected)
report = classification_report(y_test, y_pred)
print(report)
#Genetic Algorithm+logistic regression model
selector_log_reg = GeneticSelectionCV(estimator=log_reg,
                                      cv=5,
                                      verbose=1,
                                      scoring="f1",
                                      max_features=50,
                                      n_population=50,
                                      crossover_proba=0.5,
                                      mutation_proba=0.2,
                                      n_generations=40,
                                      crossover_independent_proba=0.5,
                                      mutation_independent_proba=0.05,
                                      tournament_size=3,
                                      n_gen_no_change=10,
                                      caching=True,
                                      n_jobs=-1)
selector_log_reg.fit(X_sample, y_sample)
X_train_selected = X_train.loc[:, selector_log_reg.support_]
X_test_selected = X_test.loc[:, selector_log_reg.support_]
log_reg.fit(X_train_selected, y_train)
log_reg.fit(X_train_selected, y_train)
y_pred = log_reg.predict(X_test_selected)
report = classification_report(y_test, y_pred)
print(report)