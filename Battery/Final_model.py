import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

p = 'IRIV'
file_path = rf"{p}_1st+SG_training data.xlsx"
df_train = pd.read_excel(file_path, header=0)

val_file_path = rf"{p}_1st+SG_val data.xlsx"
df_val = pd.read_excel(val_file_path, header=0)

test_file_path = rf"{p}_1st+SG_testing data.xlsx"
df_test = pd.read_excel(test_file_path, header=0)

X_train = df_train.iloc[:, :-1].values
y_train = df_train.iloc[:, -1].values
X_val = df_val.iloc[:, :-1].values
y_val = df_val.iloc[:, -1].values
X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values

rr_params = {'alpha': 3.59E-6}
pls_params = {'scale': False, 'max_iter': 10000, 'n_components': 10}
krr_params = {'kernel': 'polynomial', 'alpha': 5.46E-8, 'coef0': 1, 'degree': 2}

pls = PLSRegression(**pls_params)
krr = KernelRidge(**krr_params)
rr = Ridge(**rr_params)

pls.fit(X_train, y_train)
krr.fit(X_train, y_train)
rr.fit(X_train, y_train)

pls_val_pred = pls.predict(X_val).ravel()
krr_val_pred = krr.predict(X_val).ravel()
rr_val_pred = rr.predict(X_val).ravel()

X_val_blend = np.column_stack([pls_val_pred, krr_val_pred, rr_val_pred])

meta_model = LinearRegression()
meta_model.fit(X_val_blend, y_val)

y_val_pred_blend = meta_model.predict(X_val_blend)
val_r2 = r2_score(y_val, y_val_pred_blend)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_blend))

pls_test_pred = pls.predict(X_test).ravel()
krr_test_pred = krr.predict(X_test).ravel()
rr_test_pred = rr.predict(X_test).ravel()

X_test_blend = np.column_stack([pls_test_pred, krr_test_pred, rr_test_pred])
y_test_pred = meta_model.predict(X_test_blend)

test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

from scipy.signal import savgol_filter
def first_derivative(spectrum):
    return np.gradient(spectrum)

final_model = meta_model

truedata_path = r'CuLiMetal.xlsx'
truedata_df = pd.read_excel(truedata_path, header=0)
true_spectrum = truedata_df.values

true_spectrum_1st = np.apply_along_axis(first_derivative,
                                        axis=1,
                                        arr=true_spectrum
                                        )

window_length = 15
polyorder = 2
true_spectrum_1st_SG = np.apply_along_axis(savgol_filter,
                                          axis=1,
                                          arr=true_spectrum_1st,
                                          window_length=window_length,
                                          polyorder=polyorder
                                          )

true_spectrum_df = pd.DataFrame(true_spectrum_1st_SG, columns=truedata_df.columns)
waves = df_train.columns[:-1]
iriv_true_spectrum_df = true_spectrum_df.loc[:, waves]

rr_true_pred = rr.predict(iriv_true_spectrum_df)
krr_true_pred = krr.predict(iriv_true_spectrum_df)
pls_true_pred = pls.predict(iriv_true_spectrum_df)

true_blend = np.column_stack([pls_true_pred, krr_true_pred, rr_true_pred])

true_results = final_model.predict(true_blend)

print(true_results)
