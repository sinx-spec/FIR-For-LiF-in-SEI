import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

p = 'SPA'
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

alpha_range = np.logspace(-1, -15, 100)
len_alpha = len(alpha_range)
coef0_range = np.array([i for i in range(21)])
len_coef0 = len(coef0_range)

rmse_grid = np.zeros((len(alpha_range), len(coef0_range)))
for i, alpha in enumerate(alpha_range):
    for j, coef0 in enumerate(coef0_range):
        print(f'当前进度({i}/{len_alpha}，{j}/{len_coef0})。')
        krr = KernelRidge(kernel='polynomial',alpha=alpha,coef0=coef0,degree=2)
        krr.fit(X_train, y_train)
        y_val_pred = krr.predict(X_val)
        rmse_grid[i,j] = np.sqrt(mean_squared_error(y_val, y_val_pred))

best_idx = np.unravel_index(np.argmin(rmse_grid), rmse_grid.shape)
print(f'最优alpha参数为{alpha_range[best_idx[0]]}',
      f'最优coef0参数为{coef0_range[best_idx[1]]}',
      f'其RMSE为{rmse_grid[best_idx]:.3f}')

plt.figure(figsize=(10, 6))
plt.imshow(rmse_grid,cmap='viridis_r',aspect='auto',
           extent=[coef0_range[0], coef0_range[-1],
                np.log10(alpha_range[-1]), np.log10(alpha_range[0])])
plt.colorbar(label='RMSE')
plt.xlabel('log10(coef0)')
plt.ylabel('log10(alpha)')
plt.title('Joint Tuning of alpha and coef0 (Polynomial Kernel)')
plt.scatter(np.log10(coef0_range[best_idx[1]]), np.log10(alpha_range[best_idx[0]]),
    color='red', marker='x', label=f'Best: alpha={alpha_range[best_idx[0]]}, coef0={coef0_range[best_idx[1]]}')
plt.legend()
plt.tight_layout()
plt.savefig('krr_alpha_coef0_heatmap.png', dpi=300)
plt.show()
plt.close()

process = pd.DataFrame(rmse_grid,
                       columns=coef0_range,
                       index=alpha_range)
process_filename = f'{p}_process.xlsx'
process.to_excel(process_filename)
print(f"参数寻优过程已保存到{process_filename}。")

# ========== 在测试集上评估 ==========
best_model = KernelRidge(kernel='polynomial',
                         alpha=alpha_range[best_idx[0]],
                         coef0=coef0_range[best_idx[1]],
                         degree=2)
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
plt.title('Test Set Prediction (Kernel Ridge - Polynomial)')
plt.legend()
plt.savefig('krr_test_prediction.png')
plt.show()

# ========== 保存Excel ==========
test_results = pd.DataFrame({
    'Sample_Index': np.arange(len(y_test)),
    'Actual': y_test,
    'Predicted': y_test_pred,
    'Fit_Line': fit_line
})
test_summary = pd.DataFrame({
    'Metric': ['best_alpha','best_coef0', 'Val RMSE', 'Val R2', 'Test RMSE', 'Test R2'],
    'Value': [alpha_range[best_idx[0]], coef0_range[best_idx[1]], val_rmse, val_r2, test_rmse, test_r2]
})

prediction_filename = f'{p}_prediction_results.xlsx'
with pd.ExcelWriter(f'{prediction_filename}') as writer:
    test_results.to_excel(writer, sheet_name='TestSet', index=False)
    test_summary.to_excel(writer, sheet_name='Summary', index=False)

print(f"预测结果与性能指标已保存到{prediction_filename}。")