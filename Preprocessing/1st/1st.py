import pandas as pd
import numpy as np

def first_derivative(spectrum):
    return np.gradient(spectrum)

files_name = ['Training set.xlsx','Val set.xlsx', 'Testing set.xlsx']
for i in files_name:
    file_path = rf'Dataset\{i}'
    df = pd.read_excel(file_path)
    X = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    X_derivative = np.apply_along_axis(
        first_derivative,
        axis=1,
        arr=X
    )

    df_derivative = pd.DataFrame(X_derivative, columns=df.columns[:-1])
    df_derivative['Label'] = labels
    a = i.replace('.xlsx', '')
    out_path = rf'1st\{a}_1st_data.xlsx'
    df_derivative.to_excel(out_path, index=False)
    print(f"ok")