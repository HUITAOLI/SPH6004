import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import compose
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data=pd.read_csv('sph6004_assignment1_data.csv')
#Dataset Description
# data.info()
# data.nunique(
# data.describe().transpose()
#Preprocessing
X=data.drop(['id','aki'],axis=1)
y=data['aki']
X['gcs_unable'] = X['gcs_unable'].astype('category')
numeric_data = X.select_dtypes(include=[np.number])
numeric_data.corr().round(2)
missing_ratio = X.isnull().sum() / len(X)
threshold_missing = 0.7
features_missing = missing_ratio[missing_ratio > threshold_missing].index
data_reduced =X .drop(columns=features_missing)
numeric_cols = data_reduced.select_dtypes(include=[np.number]).columns
X_numeric = data_reduced[numeric_cols]
thresholder = VarianceThreshold(threshold=0.01)
X_high_variance = thresholder.fit_transform(X_numeric)
data_high_variance = pd.DataFrame(X_high_variance, columns=numeric_cols[thresholder.get_support()])
data_precd = pd.concat([data_reduced[data_reduced.columns.difference(numeric_cols)], data_high_variance], axis=1)
corr_matrix = data_high_variance.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
threshold_corr = 0.95
corr = [column for column in upper.columns if any(upper[column] > threshold_corr)]
data_final = data_precd.drop(columns=corr)
mode_value = data_final['gcs_unable'].mode()[0]
data_final['gcs_unable'].fillna(mode_value, inplace=True)
numeric_cols = data_final.select_dtypes(include=[np.number]).columns
non_numeric_cols = data_final.select_dtypes(exclude=[np.number]).columns
numeric = data_final[numeric_cols]
imputer = KNNImputer(n_neighbors=5)
numeric_imputed = imputer.fit_transform(numeric)
numeric_imputed = pd.DataFrame(numeric_imputed, columns=numeric_cols)
non_numeric = data_final[non_numeric_cols]
imputed_data = pd.concat([non_numeric, numeric_imputed], axis=1)
imputed_data = imputed_data[data_final.columns.tolist()]
cat_cols = imputed_data.select_dtypes(exclude=[np.number]).columns
num_cols = imputed_data.select_dtypes(include=[np.number]).columns
scaler = preprocessing.StandardScaler()
oh_enc = preprocessing.OneHotEncoder( categories='auto', handle_unknown='ignore' )
preprocessor = compose.ColumnTransformer(
                transformers=[
                    ('num', scaler, num_cols),
                    ('cat', oh_enc, cat_cols),
                ], remainder="passthrough"
                )
X_preprocessed = preprocessor.fit_transform(imputed_data)
feature_names_transformed = preprocessor.get_feature_names_out()
y_binary = data['aki'].apply(lambda x: 1 if x == 3 else 0)
X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names_transformed)
X_preprocessed_df['aki'] = y_binary.values
X_preprocessed_df.to_csv('preprocessed.csv')
pca = PCA().fit(X_preprocessed)
plt.figure(figsize=(10, 7))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Different Principal Components')
plt.grid(True)
plt.show()