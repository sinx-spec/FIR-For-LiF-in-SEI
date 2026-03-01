from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_path = r"1st+SG\Training set_1st+SG_data.xlsx"
data = pd.read_excel(file_path, header=0)
X = np.array(data.iloc[:, :-1])
Y = np.array(data.iloc[:, -1])
n_samples, n_features = X.shape

def uve(m, n, rand_seed = 18):
    np.random.seed(rand_seed)
    noise = np.random.normal(loc=0, scale=0.001, size=(m, n))
    new_x = np.append(X, noise, axis=1)
    PLScoef = np.zeros((new_x.shape[0], new_x.shape[1]))
    for i in range(m):
        new_x_delete = np.delete(new_x, i, axis=0)
        y_delete = np.delete(Y, i, axis=0)
        pls = PLSRegression(n_components=38)
        pls.fit(new_x_delete, y_delete)
        PLScoef[i, :] = pls.coef_.reshape(1, -1)[0]
    meancoef = np.mean(PLScoef, axis=0)
    stdcoef = np.std(PLScoef, axis=0)
    h = meancoef / stdcoef
    h_select = h[n:]
    h_max = np.max(abs(h_select))
    index = []
    for j in range(0, n):
        if h_max <= abs(h[j]):
            index.append(j)
    selected_wave = index
    print(selected_wave)
    selected_wavelength = X[:, index]
    print(selected_wavelength.shape)

    col_list = data.columns[:-1]
    filter_features = [col_list[i] for i in selected_wave]
    uve_data = pd.DataFrame(selected_wavelength, columns=filter_features)
    uve_data['Label'] = Y
    out_path = r'UVE_1st+SG_training data.xlsx'
    uve_data.to_excel(out_path, index=False)

    plt.figure(figsize=(10, 6))
    spectrum_h = h[:n]
    noise_h = h[n:]
    spectrum_h_pd, noise_h_pd = pd.DataFrame(spectrum_h), pd.DataFrame(noise_h)
    spectrum_h_pd.to_excel('spectrum.xlsx', index=False)
    noise_h_pd.to_excel('noise.xlsx', index=False)

    plt.plot(range(n), spectrum_h, label='Spectral')
    plt.plot(range(n, n + len(noise_h)), noise_h, label='Noise')

    noise_max = np.max(noise_h)
    noise_min = np.min(noise_h)

    plt.axhline(y=noise_max, color='r', linestyle='--', label=f'Max')
    plt.axhline(y=noise_min, color='g', linestyle='--', label=f'-Max')

    plt.xlabel('Input variables                                                                             Noise variables ', fontweight='bold',fontsize=14)
    plt.ylabel('Coefficient of stabilization', fontweight='bold', fontsize=16)
    plt.legend(fontsize=28, prop={'weight': 'bold'})
    plt.xticks(fontweight='bold', fontsize=10)
    plt.yticks(fontweight='bold', fontsize=10)
    plt.savefig(r'UVE方法选取波长示意图.png')
    plt.show()
    print(r'所有数据已成功存储')

uve(n_samples, n_features)

