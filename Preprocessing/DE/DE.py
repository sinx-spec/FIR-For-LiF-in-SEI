import pandas as pd
import numpy as np
from scipy.constants import degree
from scipy.signal import detrend

files_name = ['Training set.xlsx','Val set.xlsx', 'Testing set.xlsx']
for i in files_name:
    file_path = rf'Dataset\{i}'
    df = pd.read_excel(file_path)
    X = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    X_detrended = np.apply_along_axis(
        detrend,
        axis=1,
        arr=X
    )
    df_detrended = pd.DataFrame(X_detrended, columns=df.columns[:-1])
    df_detrended['Label'] = labels
    a = i.replace('.xlsx', '')
    out_path = rf'DE\{a}_DE_data.xlsx'
    df_detrended.to_excel(out_path, index=False)
    print(f"结果已保存到 {out_path}")