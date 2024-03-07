import pandas as pd
from sklearn.metrics import  classification_report
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('preprocessed.csv',index_col=0)
X=data.drop(['aki'],axis=1)
y=data['aki']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
smote_sampler = SMOTE(random_state=12,sampling_strategy='minority')
X_train, y_train= smote_sampler.fit_resample(X_train, y_train)
selected_features=['num__admission_age', 'num__sbp_min', 'num__sbp_mean', 'num__dbp_min',
       'num__mbp_min', 'num__resp_rate_max', 'num__spo2_min', 'num__ph_min',
       'num__so2_min', 'num__so2_max', 'num__po2_min', 'num__po2_max',
       'num__pao2fio2ratio_min', 'num__baseexcess_min', 'num__baseexcess_max',
       'num__totalco2_min', 'num__hematocrit_min.1', 'num__hematocrit_max.1',
       'num__albumin_min', 'num__albumin_max', 'num__aniongap_min',
       'num__aniongap_max', 'num__bicarbonate_min.1', 'num__bun_min',
       'num__chloride_min.1', 'num__chloride_max.1', 'num__glucose_max.2',
       'num__potassium_min.1', 'num__potassium_max.1',
       'num__abs_lymphocytes_min', 'num__inr_min', 'num__inr_max',
       'num__pt_max', 'num__ptt_min', 'num__ptt_max',
       'num__bilirubin_total_min', 'num__gcs_min', 'num__gcs_motor',
       'num__gcs_verbal', 'num__gcs_eyes', 'num__weight_admit',
       'cat__gender_F', 'cat__gender_M', 'cat__race_WHITE']
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
features = ['num__hematocrit_min.1', 'num__sbp_mean']
target = 'aki'
visual = data[features + [target]].dropna()
plt.figure(figsize=(10, 8))
sns.scatterplot(x=features[0], y=features[1], hue='aki', style='aki', data=visual)
plt.title('Visualization of AKI Categories Based on Two Features')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.legend(title='AKI Category', labels=['AKI<3', 'AKI=3'])
plt.show()
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
}
stratifiedCV = StratifiedKFold(n_splits=5)
BestParams = GridSearchCV(
    estimator=SVC(kernel='rbf', random_state=42),
    param_grid=param_grid,
    cv=stratifiedCV ,
    scoring='f1',
    n_jobs=-1
)
BestParams.fit(X_train_selected, y_train)
best_svc = BestParams.best_estimator_
y_pred_test = best_svc.predict(X_test_selected)
report = classification_report(y_test, y_pred_test)
print(report)