import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder, StandardScaler

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

p = 'CARS'
train_filepath = rf"{p}\1st+SG\{p}_1st+SG_training data.xlsx"
df_train = pd.read_excel(train_filepath, header=0)
X_train = df_train.iloc[:, :-1].values
y_train = df_train.iloc[:, -1].values

val_filepath = rf'{p}\1st+SG\{p}_1st+SG_val data.xlsx'
df_val = pd.read_excel(val_filepath, header=0)
X_val = df_val.iloc[:, :-1].values
y_val = df_val.iloc[:, -1].values

test_filepath = rf"{p}\1st+SG\{p}_1st+SG_testing data.xlsx"
df_test = pd.read_excel(test_filepath, header=0)
X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values

c_arr = np.logspace(0, 5, 50)
len_c = len(c_arr)
rmse_list = []

for c_num, c in enumerate(c_arr):
    print(f'当前进度为第{c_num}/{len_c}')
    svr = SVR(C=c, epsilon=0.001, kernel='rbf')
    cross_rmse = []
    svr.fit(X_train, y_train)
    y_val_pred = svr.predict(X_val)
    rmse_list.append(np.sqrt(mean_squared_error(y_val, y_val_pred)))

best_idx = np.argmin(rmse_list)
print(f'最优参数为{c_arr[best_idx]}，其RMSE为{rmse_list[best_idx]:.3f}')

plt.figure(figsize=(8,5))
plt.plot(c_arr, rmse_list, marker='o')
plt.xscale('log')
plt.xlabel('C (log scale)')
plt.ylabel('RMSE')
plt.title('CARS SVR Hyperparameter Tuning Curve')
plt.grid()
plt.show()

# ========== 保存Excel ==========
process = pd.DataFrame({
    'C': c_arr,
    'Cross-validated RMSE': rmse_list,
})

process.to_excel(f'{p}_process.xlsx', index=False)
print("参数寻优过程已保存")

# ========== 在测试集上评估 ==========
best_model = SVR(C=c_arr[best_idx], epsilon=0.001, kernel='rbf')
best_model.fit(X_train, y_train)

y_val_pred = best_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_r2 = r2_score(y_val, y_val_pred)
print(f"验证集 RMSE: {val_rmse:.3f}")
print(f"验证集 R2: {val_r2:.3f}")

y_test_pred = best_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)
print(f"测试集 RMSE: {test_rmse:.3f}")
print(f"测试集 R2: {test_r2:.3f}")

# ========== 可视化 ==========
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_test_pred, color='blue', label='Predicted vs Actual')
slope, intercept = np.polyfit(y_test, y_test_pred, 1)
fit_line = slope * y_test + intercept
plt.plot(y_test, fit_line, color='red', label=f'Test R2: {test_r2:.3f}')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Test Set Prediction')
plt.legend()
plt.savefig('test_prediction.png')
plt.show()
plt.close()

# ========== 保存Excel ==========
test_results = pd.DataFrame({
    'Sample_Index': np.arange(len(y_test)),
    'Actual': y_test,
    'Predicted': y_test_pred,
    'Fit_Line': fit_line
})
test_summary = pd.DataFrame({
    'Metric': ['Best alpha', 'Val RMSE', 'Val R2', 'Test RMSE', 'Test R2'],
    'Value': [c_arr[best_idx], val_rmse, val_r2, test_rmse, test_r2]
})

with pd.ExcelWriter(f'{p}_prediction_results.xlsx') as writer:
    test_results.to_excel(writer, sheet_name='TestSet', index=False)
    test_summary.to_excel(writer, sheet_name='Summary', index=False)

print("预测结果与性能指标已保存")
