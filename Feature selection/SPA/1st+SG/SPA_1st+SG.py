import pandas as pd
import numpy as np
from scipy.linalg import qr, inv, pinv
import scipy.stats
import scipy.io as scio
# from progress.bar import Bar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

class SPA:

    def _projections_qr(self, X, k, M):
        X_projected = X.copy()
        norms = np.sum((X ** 2), axis=0)
        norm_max = np.amax(norms)
        X_projected[:, k] = X_projected[:, k] * 2 * norm_max / norms[k]
        _, __, order = qr(X_projected, 0, pivoting=True)

        return order[:M].T

    def _validation(self, Xcal, ycal, var_sel, Xval=None, yval=None):

        N = Xcal.shape[0]
        if Xval is None:
            NV = 0
        else:
            NV = Xval.shape[0]

        yhat = e = None

        if NV > 0:
            Xcal_ones = np.hstack(
                [np.ones((N, 1)), Xcal[:, var_sel].reshape(N, -1)])

            b = np.linalg.lstsq(Xcal_ones, ycal, rcond=None)[0]
            np_ones = np.ones((NV, 1))
            Xval_ = Xval[:, var_sel]
            X = np.hstack([np.ones((NV, 1)), Xval[:, var_sel]])
            yhat = X.dot(b)
            e = yval - yhat
        else:
            yhat = np.zeros((N, 1))
            for i in range(N):
                cal = np.hstack([np.arange(i), np.arange(i + 1, N)])
                X = Xcal[cal, var_sel.astype(np.int)]
                y = ycal[cal]
                xtest = Xcal[i, var_sel]
                X_ones = np.hstack([np.ones((N - 1, 1)), X.reshape(N - 1, -1)])
                b = np.linalg.lstsq(X_ones, y, rcond=None)[0]
                yhat[i] = np.hstack([np.ones(1), xtest]).dot(b)
            e = ycal - yhat

        return yhat, e

    def spa(self, Xcal, ycal, m_min=1, m_max=None, Xval=None, yval=None, autoscaling=1):

        assert (autoscaling == 0 or autoscaling == 1),

        N, K = Xcal.shape

        if m_max is None:
            if Xval is None:
                m_max = min(N - 1, K)
            else:
                m_max = min(N - 2, K)

        assert m_max <= min(N - (1 if Xval is not None else 2), K),

        normalization_factor = None
        if autoscaling == 1:
            normalization_factor = np.std(
                Xcal, ddof=1, axis=0).reshape(1, -1)[0]
        else:
            normalization_factor = np.ones((1, K))[0]

        Xcaln = np.empty((N, K))
        for k in range(K):
            x = Xcal[:, k]
            Xcaln[:, k] = (x - np.mean(x)) / normalization_factor[k]

        SEL = np.zeros((m_max, K))

        #with Bar('Projections :', max=K) as bar:
        for k in range(K):
            SEL[:, k] = self._projections_qr(Xcaln, k, m_max)
        #        bar.next()


        PRESS = float('inf') * np.ones((m_max + 1, K))

        #with Bar('Evaluation of variable subsets :', max=(K) * (m_max - m_min + 1)) as bar:
        for k in range(K):
            for m in range(m_min, m_max + 1):
                var_sel = SEL[:m, k].astype(int)
                _, e = self._validation(Xcal, ycal, var_sel, Xval, yval)
                PRESS[m, k] = np.conj(e).T.dot(e)
        
        #            bar.next()

        PRESSmin = np.min(PRESS, axis=0)
        m_sel = np.argmin(PRESS, axis=0)
        k_sel = np.argmin(PRESSmin)

        var_sel_phase2 = SEL[:m_sel[k_sel], k_sel].astype(int)

        Xcal2 = np.hstack([np.ones((N, 1)), Xcal[:, var_sel_phase2]])
        b = np.linalg.lstsq(Xcal2, ycal, rcond=None)[0]
        std_deviation = np.std(Xcal2, ddof=1, axis=0)

        relev = np.abs(b * std_deviation.T)
        relev = relev[1:]

        index_increasing_relev = np.argsort(relev, axis=0)
        index_decreasing_relev = index_increasing_relev[::-1].reshape(1, -1)[0]

        PRESS_scree = np.empty(len(var_sel_phase2))
        yhat = e = None
        for i in range(len(var_sel_phase2)):
            var_sel = var_sel_phase2[index_decreasing_relev[:i + 1]]
            _, e = self._validation(Xcal, ycal, var_sel, Xval, yval)

            PRESS_scree[i] = np.conj(e).T.dot(e)

        RMSEP_scree = np.sqrt(PRESS_scree / len(e))

        PRESS_scree_min = np.min(PRESS_scree)
        alpha = 0.05
        dof = len(e)
        fcrit = scipy.stats.f.ppf(1 - alpha, dof, dof)
        PRESS_crit = PRESS_scree_min * fcrit


        i_crit = np.min(np.nonzero(PRESS_scree < PRESS_crit))
        i_crit = max(m_min, i_crit)

        var_sel = var_sel_phase2[index_decreasing_relev[:i_crit]]

        SPA_tuning = pd.DataFrame({
            'i_crit' : range(1, len(RMSEP_scree)+1),
            'RMSEP_scree' : RMSEP_scree
        })
        SPA_tuning.to_excel('SPA_tuning.xlsx', index=False)

        var_sel_pd = pd.DataFrame(var_sel)
        var_sel_pd.to_excel('var_sel.xlsx', index=False)

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure()
        plt.xlabel('Number of variables included in the model')
        plt.ylabel('RMSE')
        plt.title('Final number of selected variables:{}(RMSE={})'.format(len(var_sel), RMSEP_scree[i_crit]))
        plt.plot(RMSEP_scree)
        plt.scatter(i_crit, RMSEP_scree[i_crit], marker='s', color='r')
        plt.grid(True)

        plt.figure()
        plt.plot(Xcal[0, :])
        plt.scatter(var_sel, Xcal[0, var_sel], marker='s', color='r')
        plt.legend(['First calibration object', 'Selected variables'])
        plt.xlabel('Variable index')
        plt.grid(True)
        plt.show()

        return var_sel, var_sel_phase2

    def __repr__(self):
        return "SPA()"


if __name__ == "__main__":
    file_path = r"1st+SG\Training set_1st+SG_data.xlsx"
    df = pd.read_excel(file_path, header=0)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    Xcal, Xval, ycal, yval = train_test_split(X, y, test_size=0.2, random_state=66)

    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

    Xcal = min_max_scaler.fit_transform(Xcal)
    Xval = min_max_scaler.transform(Xval)
   
    var_sel, var_sel_phase2 = SPA().spa(
        Xcal, ycal, m_min=2, m_max=251, Xval=Xval, yval=yval, autoscaling=1)

index_file = r'var_sel.xlsx'
de_train_file = r'1st+SG\Training set_1st+SG_data.xlsx'
de_val_file = r'1st+SG\Val set_1st+SG_data.xlsx'
de_test_file = r'1st+SG\Testing set_1st+SG_data.xlsx'

df_index = pd.read_excel(index_file, header=0)
index_list = []
for idx in df_index.values:
    index_list.extend(idx)
index_list.sort()
index_list.append(-1)

df_de_train = pd.read_excel(de_train_file, header=0)
df_de_val = pd.read_excel(de_train_file, header=0)
df_de_test = pd.read_excel(de_test_file, header=0)

df_SPA_train = df_de_train.iloc[:, index_list]
df_SPA_val = df_de_val.iloc[:, index_list]
df_SPA_test = df_de_test.iloc[:, index_list]

df_SPA_train.to_excel('SPA_1st+SG_training data.xlsx', index=False)
df_SPA_val.to_excel('SPA_1st+SG_val data.xlsx', index=False)
df_SPA_test.to_excel('SPA_1st+SG_testing data.xlsx', index=False)

print('ok')