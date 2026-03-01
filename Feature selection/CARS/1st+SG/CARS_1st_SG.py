import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from typing import Optional, List

class CARS:
    def __init__(self, X: np.ndarray, y: np.ndarray, n_iter: int = 50,
                 n_end: int = 2, n_components_max: int = 100, cv: int = 5,
                 sample_ratio: float = 0.8, random_state: Optional[int] = 42,
                 verbose: bool = True):
        self.X = np.asarray(X)
        self.y = np.asarray(y).reshape(-1)
        self.n_samples, self.n_features = self.X.shape
        self.n_iter = n_iter
        self.n_end = max(1, n_end)
        self.n_components_max = max(1, n_components_max)
        self.cv = cv
        self.sample_ratio = sample_ratio
        self.verbose = verbose
        self.rng = np.random.default_rng(random_state)

        Nvar, Nend = self.n_features, self.n_end
        if self.n_iter == 1:
            self.r_seq = [Nend]
        else:
            self.r_seq = [
                max(1, int(round(Nvar * (Nend / Nvar) ** (i / (self.n_iter - 1)))))
                for i in range(self.n_iter)
            ]
            for i in range(1, len(self.r_seq)):
                if self.r_seq[i] > self.r_seq[i-1]:
                    self.r_seq[i] = self.r_seq[i-1]

        self.rmsecv_history: List[float] = []
        self.num_vars_history: List[int] = []
        self.selected_sets: List[np.ndarray] = []
        self.coef_history: List[np.ndarray] = []
        self.best_iter: Optional[int] = None
        self.best_subset_: Optional[np.ndarray] = None
        self.freq_: Optional[np.ndarray] = None

    def _rmse_from_neg_mse(self, scores: np.ndarray) -> float:
        return float(np.sqrt(np.mean(-scores)))

    def _cv_opt_ncomp(self, X_sub: np.ndarray, y_sub: np.ndarray, max_comp: int) -> int:
        best_c, best_rmse = 1, np.inf
        for c in range(1, max_comp + 1):
            pls = PLSRegression(n_components=c)
            scores = cross_val_score(pls, X_sub, y_sub, cv=self.cv, scoring="neg_mean_squared_error")
            rmse = self._rmse_from_neg_mse(scores)
            if rmse < best_rmse:
                best_rmse, best_c = rmse, c
        return best_c

    def fit(self, df: pd.DataFrame, label_col: Optional[str] = None):
        current_vars = np.arange(self.n_features)
        freq = np.zeros(self.n_features, dtype=int)

        for it in range(self.n_iter):
            r_target = self.r_seq[it]
            n_cal = max(2, int(round(self.sample_ratio * self.n_samples)))
            id_cal = self.rng.choice(self.n_samples, size=n_cal, replace=False)

            X_cur = self.X[id_cal][:, current_vars]
            y_cur = self.y[id_cal]

            c0 = min(min(X_cur.shape)-1, self.n_components_max)
            c0 = max(1, c0)
            pls0 = PLSRegression(n_components=c0)
            pls0.fit(X_cur, y_cur)

            w = np.abs(pls0.coef_).reshape(-1)
            if np.allclose(w.sum(), 0):
                w = np.ones_like(w)/len(w)
            else:
                w /= w.sum()

            if r_target >= len(current_vars):
                next_vars = current_vars.copy()
            else:
                u = self.rng.random(len(current_vars))
                keys = u ** (1.0 / (w + 1e-12))
                sel_idx = np.argpartition(keys, r_target - 1)[:r_target]
                sel_idx = sel_idx[np.argsort(keys[sel_idx])]
                next_vars = current_vars[sel_idx]

            X_sel = self.X[id_cal][:, next_vars]
            y_sel = self.y[id_cal]
            max_c = min(self.n_components_max, X_sel.shape[0]-1, X_sel.shape[1])
            max_c = max(1, max_c)
            best_c = self._cv_opt_ncomp(X_sel, y_sel, max_c)

            pls_final = PLSRegression(n_components=best_c)
            pls_final.fit(X_sel, y_sel)
            scores = cross_val_score(pls_final, X_sel, y_sel, cv=self.cv, scoring="neg_mean_squared_error")
            rmse = self._rmse_from_neg_mse(scores)

            self.rmsecv_history.append(rmse)
            self.num_vars_history.append(len(next_vars))
            self.selected_sets.append(next_vars)
            coef_full = np.zeros(self.n_features)
            coef_full[next_vars] = pls_final.coef_.reshape(-1)
            self.coef_history.append(coef_full)
            freq[next_vars] += 1

            if self.verbose:
                print(f"Iter {it+1:02d}/{self.n_iter} | vars={len(next_vars):4d} | best_c={best_c:2d} | RMSECV={rmse:.4f}")

            current_vars = next_vars

        self.best_iter = int(np.argmin(self.rmsecv_history))
        self.best_subset_ = self.selected_sets[self.best_iter]
        self.freq_ = freq

        W_list = sorted(list(self.best_subset_))
        col_list = df.columns[:-1] if label_col is None else [c for c in df.columns if c != label_col]
        filter_features = [col_list[i] for i in W_list]
        X_arr = np.array(df[col_list])
        Y_arr = np.array(df[label_col] if label_col else df.iloc[:, -1])
        cars_data = pd.DataFrame(X_arr[:, W_list], columns=filter_features)
        cars_data['Label'] = Y_arr
        cars_data.to_excel('CARS_1st+SG_training data.xlsx', index=False)

        selection_matrix = np.array([np.isin(range(self.n_features), s, assume_unique=True) for s in self.selected_sets], dtype=int)
        retention_count = selection_matrix.sum(axis=0)
        retention_rate = retention_count / self.n_iter

        with pd.ExcelWriter("CARS_tuning.xlsx") as writer:
            pd.DataFrame({
                "Iteration": range(1, self.n_iter+1),
                "RMSECV": self.rmsecv_history,
                "Num_vars": self.num_vars_history
            }).to_excel(writer, sheet_name="RMSECV_history", index=False)

            pd.DataFrame({
                "Variable": col_list,
                "Retention_count": retention_count.astype(int),
                "Retention_rate": retention_rate
            }).to_excel(writer, sheet_name="Variable_retention", index=False)

            coef_df = pd.DataFrame(self.coef_history, columns=col_list)
            coef_df.insert(0, "Iteration", range(1, self.n_iter+1))
            coef_df.to_excel(writer, sheet_name="Coef_history", index=False)

        return self

    def plot_diagnostics(self):
        iters = np.arange(1, self.n_iter + 1)
        plt.figure()
        plt.plot(iters, self.rmsecv_history, marker='o')
        plt.scatter(self.best_iter + 1, self.rmsecv_history[self.best_iter], s=80)
        plt.xlabel('Iteration'); plt.ylabel('RMSECV')
        plt.title('CARS — RMSECV across iterations')
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(iters, self.num_vars_history, marker='o')
        plt.scatter(self.best_iter + 1, self.num_vars_history[self.best_iter], s=80)
        plt.xlabel('Iteration'); plt.ylabel('Number of variables')
        plt.title('CARS — variable count across iterations')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    file_path = r"1st+SG\Training set_1st+SG_data.xlsx"
    df = pd.read_excel(file_path, header=0)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    cars = CARS(
        X, y,
        n_iter=10,
        n_end=2,
        n_components_max=60,
        cv=5,
        sample_ratio=0.4,
        random_state=68 ,
        verbose=True,
    )
    cars.fit(df=df)
    print("最佳波长索引:", cars.best_subset_)
    cars.plot_diagnostics()




