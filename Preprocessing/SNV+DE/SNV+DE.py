import pandas as pd
import numpy as np
from scipy.signal import detrend

def snv(input_data):
    mean = np.mean(input_data)
    std = np.std(input_data)
    return (input_data - mean) / std

files_name = ['Training set.xlsx','Val set.xlsx', 'Testing set.xlsx']
for i in files_name:
    file_path = rf'Dataset\{i}'
    df = pd.read_excel(file_path)
    X = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    X_snv = np.apply_along_axis(
        snv,
        axis=1,
        arr=X
    )
    X_snv_de = np.apply_along_axis(
        detrend,
        axis=1,
        arr=X_snv
    )
    df_detrended = pd.DataFrame(X_snv_de, columns=df.columns[:-1])
    df_detrended['Label'] = labels
    a = i.replace('.xlsx', '')
    out_path = rf'SNV+DE\{a}_SNV+DE_data.xlsx'
    df_detrended.to_excel(out_path, index=False)
    print(f"结果已保存到 {out_path}")
