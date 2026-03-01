import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

def first_derivative(spectrum):
    return np.gradient(spectrum)

files_name = ['Training set.xlsx','Val set.xlsx', 'Testing set.xlsx']
for i in files_name:
    file_path = rf'Dataset\{i}'
    df = pd.read_excel(file_path)
    X = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    X_1st = np.apply_along_axis(
        first_derivative,
        axis=1,
        arr=X
    )
    window_length = 15
    polyorder = 2
    X_1st_SG = np.apply_along_axis(
        savgol_filter,
        axis=1,
        arr=X_1st,
        window_length=window_length,
        polyorder=polyorder
    )

    df_1st_SG = pd.DataFrame(X_1st_SG, columns=df.columns[:-1])
    df_1st_SG['Label'] = labels
    a = i.replace('.xlsx', '')
    out_path = rf'1st+SG\{a}_1st+SG_data.xlsx'
    df_1st_SG.to_excel(out_path, index=False)
    print(f"结果已保存到 {out_path}")
