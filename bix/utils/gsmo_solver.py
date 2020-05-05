import numpy as np
from scipy.linalg import null_space


class GSMO:
    def __init__(self, A, b, C, d, r, R, optimization_type):
        # optimize F: x'Ax + b'x  s.t.  Cx=d, x elements [r,R]^n
        self.A = A
        self.b = b
        self.C = C
        self.d = d
        # lower bound
        self.r = r
        # upper bound
        self.R = R
        # number of components
        self.n = A.shape[0]
        # minimize or maximize
        self.optimization_type = optimization_type
        # size of working set
        self.K = np.linalg.matrix_rank(C) + 1
        # TODO: first guess such that Cx = d and x elements [r,R]^n
        self.x = np.empty((1, self.n))
        # initial gradient
        self.gradient = (self.A + self.A.transpose()).dot(self.x) + self.b

    # first K - 1 Elements
    def __init_working_set(self, x):
        S = []
        S_a = []
        S_i = []
        v = np.empty((1, self.n), dtype=[('idx', int), ('val', float)])
        for i in range(self.n):
            w_best = self.__find_optimal_gradient_displacement(x[i], self.gradient[i])
            v[i] = (i, abs((w_best - x[i]) * self.gradient[i]))

            if not v[i] == 0:
                S_a.append(i)
            else:
                S_i.append(i)
        if len(S_a) > self.K:
            p = round(len(S_a) * 0.1)
            sorted_v = np.sort(v, order='val')
            for i in range(self.K - p - 1):
                S.append(sorted_v[i][1])

            intersection = np.setdiff1d(S_a, S)
            S.extend(np.random.choice(intersection, p))

        else:
            S.extend(S_a)
            random_idx_count = self.K - 1 - len(S_a)
            S.extend(np.random.choice(S_i, random_idx_count))

        return S

    def __find_optimal_gradient_displacement(self, x_i, df_i):
        choice_r = (self.r - x_i) * df_i
        choice_R = (self.R - x_i) * df_i
        if self.optimization_type == 'maximize':
            # QP is maximized we pick n_i* such that n_i* x df_i >= 0
            # n_i = (w - x_i)
            if choice_r >= choice_R:
                return self.r
            else:
                return self.R
        else:
            # OP is minimized we pick n_i* such that n_i* x df_i <= 0
            # n_i = (w - x_i)
            if choice_r <= choice_R:
                return self.r
            else:
                return self.R

    def __solve_small_QP(self, x, dF, A, C, S):
        u_k = null_space(C[:, S])
        a_k = self.__find_optimal_solution(x, dF, A, C, S)
        return a_k * u_k

    def __find_optimal_solution(self, x, dF, A, C, S):
        D = self.K - np.linalg.matrix_rank(C[:, S])
        bounds = []
        for i in range(D):
            bounds.append(self.__get_bounds(i, x, C[:, S]))
        bounds = tuple(bounds)

    def __get_bounds(self, ):
        pass




