import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
import os
from sklearn.metrics import mean_squared_error, r2_score

file_path = r"1st+SG\Training set_1st+SG_data.xlsx"
df = pd.read_excel(file_path, header=0)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
feature_names = list(df.columns[:-1])

class Elimination:
    def __init__(self, max_components=130):
        self.max_components = max_components

    def iteration(self, data, label):
        delete_index = []
        base_score = self._get_score(data, label)
        while True:
            scores = self._get_partial_score(data, label, delete_index)
            if base_score < scores.min():
                break
            else:
                base_score = scores.min()
                index = np.argmin(scores)
                delete_index.append(index)
        return np.delete(data, delete_index, axis=1), delete_index

    def _get_partial_score(self, data, label, delete_index):
        scores = []
        for i in range(data.shape[1]):
            if i in delete_index:
                scores.append(np.inf)
                continue
            idx = delete_index.copy()
            idx.append(i)
            sub_data = np.delete(data, idx, axis=1)
            n_component = min(sub_data.shape[1], self.max_components)
            model = PLSRegression(n_components=n_component, max_iter=1000)
            score = cross_val_score(model, sub_data, label, cv=5, n_jobs=-1,
                                    scoring='neg_mean_squared_error').mean()
            scores.append(np.sqrt(-score))
        return np.stack(scores)

    def _get_score(self, data, label):
        n_component = min(data.shape[1], self.max_components)
        model = PLSRegression(n_components=n_component, max_iter=1000)
        score = cross_val_score(model, data, label, cv=5, n_jobs=-1, scoring='neg_mean_squared_error').mean()
        return np.sqrt(-score)

class IRIV:
    def __init__(self, max_components=130, random_state=None):
        self.max_components = max_components
        self.back_elimination = Elimination(max_components)
        self.variable_counts = []
        self.random_state = random_state

        self.all_iterations_data = []
        self.current_iteration_data = {}

        self.feature_names = None
        self.original_feature_names = None

    def iteration(self, data, label, iter_num=100, min_dimension=50, feature_names=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        self.data, self.label = data, label
        self.original_feature_names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(data.shape[1])]
        self.feature_names = self.original_feature_names.copy()

        self.original_indices = list(range(data.shape[1]))

        for j in range(iter_num):
            start_time = time.time()
            store_variables, remove_variables = self._calculate_informative_variable(data, label)
            self.variable_counts.append(int(np.sum(store_variables)))

            iteration_data = {
                'iteration': j + 1,
                'strong_vars': [self.original_indices[i] for i in self.current_iteration_data.get('strong_vars', [])],
                'weak_vars': [self.original_indices[i] for i in self.current_iteration_data.get('weak_vars', [])],
                'uninform_vars': [self.original_indices[i] for i in
                                  self.current_iteration_data.get('uninform_vars', [])],
                'interfering_vars': [self.original_indices[i] for i in
                                     self.current_iteration_data.get('interfering_vars', [])],
                'rmsecv_exclude_all': {self.original_indices[i]: v for i, v in
                                       self.current_iteration_data.get('rmsecv_exclude_all', {}).items()},
                'rmsecv_include_all': {self.original_indices[i]: v for i, v in
                                       self.current_iteration_data.get('rmsecv_include_all', {}).items()}
            }
            self.all_iterations_data.append(iteration_data)

            if np.sum(remove_variables) == 0 or np.sum(store_variables) <= min_dimension:
                data = data[:, store_variables]
                self.feature_names = [self.feature_names[i] for i, keep in enumerate(store_variables) if keep]
                self.original_indices = [self.original_indices[i] for i, keep in enumerate(store_variables) if keep]
                break

            data = data[:, store_variables]
            self.feature_names = [self.feature_names[i] for i, keep in enumerate(store_variables) if keep]
            self.original_indices = [self.original_indices[i] for i, keep in enumerate(store_variables) if keep]

        data_after, _ = self.back_elimination.iteration(data, label)
        self.remain_data = data_after


        self.final_indices = self.original_indices

        return data_after

    def _calculate_informative_variable(self, data, label):
        rmsecv5, A = self._calculate_rmsecv(data, label)
        rmsecv_origin = np.tile(rmsecv5[:, 0].reshape(-1, 1), (A.shape[1],))
        rmsecv_replace = rmsecv5[:, 1:]

        rmsecv_exclude = rmsecv_replace.copy()
        rmsecv_include = rmsecv_replace.copy()
        rmsecv_exclude[A == 0] = rmsecv_origin[A == 0]
        rmsecv_include[A == 1] = rmsecv_origin[A == 1]

        self.current_iteration_data['rmsecv_exclude_all'] = {}
        self.current_iteration_data['rmsecv_include_all'] = {}
        for i in range(A.shape[1]):
            self.current_iteration_data['rmsecv_exclude_all'][i] = rmsecv_exclude[:, i]
            self.current_iteration_data['rmsecv_include_all'][i] = rmsecv_include[:, i]

        exclude_mean = np.mean(rmsecv_exclude, axis=0)
        include_mean = np.mean(rmsecv_include, axis=0)

        p_val, DMEAN, H = [], [], []
        for i in range(A.shape[1]):
            _, pVal = stats.mannwhitneyu(rmsecv_exclude[:, i], rmsecv_include[:, i], alternative='two-sided')
            H.append(int(pVal <= 0.05))
            temp_DMEAN = exclude_mean[i] - include_mean[i]
            if temp_DMEAN < 0:
                pVal = pVal + 1
            p_val.append(pVal)
            DMEAN.append(temp_DMEAN)

        p_val = np.stack(p_val)
        H = np.stack(H)

        strong_inform = (H == 1) & (p_val < 1)
        weak_inform = (H == 0) & (p_val < 1)
        un_inform = (H == 0) & (p_val >= 1)
        interfering = (H == 1) & (p_val >= 1)

        self.current_iteration_data['strong_vars'] = list(np.where(strong_inform)[0])
        self.current_iteration_data['weak_vars'] = list(np.where(weak_inform)[0])
        self.current_iteration_data['uninform_vars'] = list(np.where(un_inform)[0])
        self.current_iteration_data['interfering_vars'] = list(np.where(interfering)[0])

        remove_variables = un_inform | interfering
        store_variables = strong_inform | weak_inform
        return store_variables, remove_variables

    def _calculate_rmsecv(self, data, label):
        A, row = self._generate_binary_matrix(data.shape[1])

        rmsecv5 = np.zeros((row, data.shape[1] + 1))

        for k, sub_a in enumerate(A):
            sub_data = data[:, sub_a == 1]
            n_component = min(np.sum(sub_a == 1), self.max_components)
            model = PLSRegression(n_components=n_component, max_iter=1000)
            score = cross_val_score(model, sub_data, label, cv=5, n_jobs=-1,
                                    scoring='neg_mean_squared_error').mean()
            rmsecv5[k, 0] = np.sqrt(-score)

        for i in range(data.shape[1]):
            B = np.copy(A)
            B[:, i] = 1 - B[:, i]
            for k, sub_b in enumerate(B):
                sub_data = data[:, sub_b == 1]
                n_component = min(np.sum(sub_b == 1), self.max_components)
                model = PLSRegression(n_components=n_component, max_iter=1000)
                score = cross_val_score(model, sub_data, label, cv=5, n_jobs=-1,
                                        scoring='neg_mean_squared_error').mean()
                rmsecv5[k, i + 1] = np.sqrt(-score)
        return rmsecv5, A

    def _generate_binary_matrix(self, n_features):
        if n_features >= 500:
            row = 500
        elif n_features >= 300:
            row = 300
        elif n_features >= 100:
            row = 200
        elif n_features >= 50:
            row = 100
        else:
            row = 50

        A = np.ones((row, n_features))
        A[row // 2:] = 0

        while True:
            A = np.stack([np.random.permutation(sub_a) for sub_a in A.T]).T
            if not np.sum(np.sum(A, axis=1) == 0):
                break
        return A, row

    def remain_index(self):
        return self.final_indices

    def _kde_curve(self, arr, num=200, extend=0.01):
        kde = stats.gaussian_kde(arr)
        xs = np.linspace(np.min(arr) - extend, np.max(arr) + extend, num=num)
        ys = kde(xs)
        return xs, ys

    def plot_variable_distribution(self, var_index, iteration=None, save_dir=None, dpi=120):
        if iteration is None:
            iteration_data = self.all_iterations_data[-1]
        else:
            iteration_data = next((item for item in self.all_iterations_data if item['iteration'] == iteration), None)
            if iteration_data is None:
                raise ValueError(f"迭代 {iteration} 的数据不存在")

        if var_index not in iteration_data['rmsecv_exclude_all']:
            raise ValueError(f"变量 {var_index} 在迭代 {iteration_data['iteration']} 中不存在")

        rmsecv_ex = iteration_data['rmsecv_exclude_all'][var_index]
        rmsecv_in = iteration_data['rmsecv_include_all'][var_index]

        plt.figure(figsize=(8, 6))
        plt.hist(rmsecv_ex, bins=20, density=True, alpha=0.4, label=f"Exclude (mean={np.mean(rmsecv_ex):.3f})")
        plt.hist(rmsecv_in, bins=20, density=True, alpha=0.4, label=f"Include (mean={np.mean(rmsecv_in):.3f})")

        try:
            xs_ex, ys_ex = self._kde_curve(rmsecv_ex, extend=0.01)
            xs_in, ys_in = self._kde_curve(rmsecv_in, extend=0.01)
            plt.plot(xs_ex, ys_ex, linewidth=2, color='blue')
            plt.plot(xs_in, ys_in, linewidth=2, color='orange')
        except Exception:
            pass

        plt.axvline(np.mean(rmsecv_ex), linestyle="--", linewidth=2, color='blue')
        plt.axvline(np.mean(rmsecv_in), linestyle="--", linewidth=2, color='orange')

        title = f"Variable {var_index} (Iteration {iteration_data['iteration']})"
        if self.original_feature_names and 0 <= var_index < len(self.original_feature_names):
            title += f" ({self.original_feature_names[var_index]})"

        category = "Unknown"
        if var_index in iteration_data['strong_vars']:
            category = "Strong informative"
        elif var_index in iteration_data['weak_vars']:
            category = "Weak informative"
        elif var_index in iteration_data['uninform_vars']:
            category = "Uninformative"
        elif var_index in iteration_data['interfering_vars']:
            category = "Interfering"

        title += f" - {category}"

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"IRIV_var_{var_index}_iter{iteration_data['iteration']}.png")
            plt.savefig(out, dpi=dpi)

    def plot_all_variable_examples(self, n_each=1, iteration=None, save_dir=None, dpi=120):
        if iteration is None:
            iteration_data = self.all_iterations_data[-1]
        else:
            iteration_data = next((item for item in self.all_iterations_data if item['iteration'] == iteration), None)
            if iteration_data is None:
                raise ValueError(f"迭代 {iteration} 的数据不存在")

        categories = {
            "Strong informative": iteration_data['strong_vars'],
            "Weak informative": iteration_data['weak_vars'],
            "Uninformative": iteration_data['uninform_vars'],
            "Interfering": iteration_data['interfering_vars']
        }

        for cat_name, var_list in categories.items():
            if len(var_list) == 0:
                print(f"No variables found in {cat_name} for iteration {iteration_data['iteration']}")
                continue
            pick = random.sample(var_list, min(n_each, len(var_list)))
            for v in pick:
                print(f"绘制 {cat_name} 变量: {v} (Iteration {iteration_data['iteration']})")
                self.plot_variable_distribution(v, iteration_data['iteration'], save_dir=save_dir, dpi=dpi)

    def save_all_variable_example_data(self, n_each=1, iteration=None, out_path='IRIV_var_examples_all.xlsx'):
        if iteration is None:
            iteration_data = self.all_iterations_data[-1]
        else:
            iteration_data = next((item for item in self.all_iterations_data if item['iteration'] == iteration), None)
            if iteration_data is None:
                raise ValueError(f"迭代 {iteration} 的数据不存在")

        categories = {
            "Strong informative": iteration_data['strong_vars'],
            "Weak informative": iteration_data['weak_vars'],
            "Uninformative": iteration_data['uninform_vars'],
            "Interfering": iteration_data['interfering_vars']
        }

        all_data = []

        for cat_name, var_list in categories.items():
            if len(var_list) == 0:
                continue
            pick = random.sample(var_list, min(n_each, len(var_list)))
            for v in pick:
                rmsecv_ex = iteration_data['rmsecv_exclude_all'][v]
                rmsecv_in = iteration_data['rmsecv_include_all'][v]
                df_ex = pd.DataFrame({
                    'Iteration': [iteration_data['iteration']] * len(rmsecv_ex),
                    'Category': [cat_name] * len(rmsecv_ex),
                    'Variable_index': [v] * len(rmsecv_ex),
                    'Variable_name': [self.original_feature_names[v] if v < len(
                        self.original_feature_names) else f"f{v}"] * len(rmsecv_ex),
                    'Type': ['Exclude'] * len(rmsecv_ex),
                    'RMSECV': rmsecv_ex
                })
                df_in = pd.DataFrame({
                    'Iteration': [iteration_data['iteration']] * len(rmsecv_in),
                    'Category': [cat_name] * len(rmsecv_in),
                    'Variable_index': [v] * len(rmsecv_in),
                    'Variable_name': [self.original_feature_names[v] if v < len(
                        self.original_feature_names) else f"f{v}"] * len(rmsecv_in),
                    'Type': ['Include'] * len(rmsecv_in),
                    'RMSECV': rmsecv_in
                })
                all_data.append(df_ex)
                all_data.append(df_in)

        if all_data:
            df_all = pd.concat(all_data, ignore_index=True)
            df_all.to_excel(out_path, index=False)
            print(f"RMSECV example data saved to {out_path}")


def calculate_retained_index(train_data, train_label, feature_names=None, random_state=0):
    start_time = time.time()
    iriv_model = IRIV(max_components=38, random_state=random_state)
    _ = iriv_model.iteration(train_data, train_label, feature_names=feature_names)
    retained_index = iriv_model.remain_index()
    end_time = time.time()

    print('计算强弱信息变量完成，耗时 %.3fs' % (end_time - start_time))
    print('保留索引[从0开始]：', retained_index)
    print('最终保留变量数量：', len(retained_index))
    print('折线图显示的最终保留数量：', iriv_model.variable_counts[-1])

    retained_index_list = sorted(list(retained_index))
    col_list = df.columns[:-1]
    filter_features = [col_list[i] for i in retained_index_list]
    X_arr = np.array(df.iloc[:, :-1])
    Y_arr = np.array(df.iloc[:, -1])
    iriv_data = pd.DataFrame(X_arr[:, retained_index_list], columns=filter_features)
    iriv_data['Label'] = Y_arr
    out_path = r'IRIV_1st+SG_training data.xlsx'
    iriv_data.to_excel(out_path, index=False)

    iterations = range(1, len(iriv_model.variable_counts) + 1)
    counts = iriv_model.variable_counts
    IRIV_tuning_df = pd.DataFrame({'iterations': list(iterations), 'counts': counts})
    IRIV_tuning_df.to_excel('IRIV_tuning.xlsx', index=False)

    return retained_index, iriv_model

retained_index, iriv_model = calculate_retained_index(X, y, feature_names=feature_names)

for iteration_data in iriv_model.all_iterations_data:
    iteration = iteration_data['iteration']

    iter_dir = os.path.join('IRIV_var_examples', f'iteration_{iteration}')
    os.makedirs(iter_dir, exist_ok=True)

    iriv_model.plot_all_variable_examples(
        n_each=1,
        iteration=iteration,
        save_dir=iter_dir,
        dpi=150
    )

    out_path = f'IRIV_var_examples_data_iter{iteration}.xlsx'
    iriv_model.save_all_variable_example_data(
        n_each=1,
        iteration=iteration,
        out_path=out_path
    )



