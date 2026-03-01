import pandas as pd
import numpy as np


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
    df_snv = pd.DataFrame(X_snv, columns=df.columns[:-1])
    df_snv['Label'] = labels
    a = i.replace('.xlsx', '')
    out_path = rf'SNV\{a}_SNV_data.xlsx'
    df_snv.to_excel(out_path, index=False)
    print(f"结果已保存到 {out_path}")