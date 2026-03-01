import pandas as pd

train_file = r'1st+SG\Training set_1st+SG_data.xlsx'
val_file =  r'1st+SG\Val set_1st+SG_data.xlsx'
test_file = r'1st+SG\Testing set_1st+SG_data.xlsx'

folder_list = ['CARS', 'UVE', 'SPA']
for alg in folder_list:
    file_path = rf'{alg}\1st+SG\{alg}_1st+SG_training data.xlsx'
    df_file = pd.read_excel(file_path, header=0)
    index_slec = df_file.columns
    df_test_file = pd.read_excel(test_file, header=0)
    df_val_file = pd.read_excel(val_file, header=0)
    df_val_slec = df_val_file.loc[:, index_slec]
    df_test_slec = df_test_file.loc[:, index_slec]
    val_out_path = rf'{alg}\1st+SG\{alg}_1st+SG_val data.xlsx'
    test_out_path = rf'{alg}\1st+SG\{alg}_1st+SG_testing data.xlsx'
    df_val_slec.to_excel(val_out_path, index=False)
    df_test_slec.to_excel(test_out_path, index=False)
    print(f'{alg} 数据集已构建')




