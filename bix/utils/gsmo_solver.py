import numpy as np
from scipy.linalg import null_space
from scipy.optimize import lsq_linear


class GSMO:
    def __init__(self, A, b, C, d, r, R, optimization_type='minimize', max_iter=1000, epsilon=0.0001, step_size=1):
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
        self.n = A.shape[1]
        # minimize or maximize
        self.optimization_type = optimization_type
        # size of working set
        self.K = np.linalg.matrix_rank(C) + 1
        # first guess such that Cx = d and x elements [r,R]^n
        result = lsq_linear(C, d, bounds=(self.r, self.R))
        self.x = result.x
        test_res = C.dot(self.x)
        if not np.allclose(d, test_res):
            raise ValueError(
                "The Equation Cx=d was not solvable. expected " + np.array_str(d) + " , got " + np.array_str(
                    test_res))
        # initial gradient
        self.gradient = (self.A + self.A.transpose()).dot(self.x) + self.b

        self.max_iter = max_iter
        self.epsilon = epsilon
        self.step_size = step_size

    def solve(self):
        for t in range(self.max_iter):
            S = self.__init_working_set()
            dF_best = 0
            j_best = -1
            dx_best_S_best = any
            j_all = [i for i in range(self.n + 1)]
            j_without_S = np.setdiff1d(j_all, S)
            for j in j_without_S:
                S.append(j)
                dx_S_best = self.__solve_small_QP(S)
                dF_temp = abs(
                    dx_S_best.transpose().dot(self.A[S, S]).dot(dx_S_best) + self.gradient[:, S].transpose().dot(
                        dx_S_best))
                if dF_temp > dF_best:
                    dF_best = dF_temp
                    j_best = j
                    dx_best_S_best = dx_S_best
                S.remove(j)

            S.append(j_best)
            self.x[S] += dx_best_S_best
            self.gradient += self.step_size * (self.A + self.A.transpose() + np.diag(self.b))[:, S].dot(dx_best_S_best)

            if dF_best < self.epsilon:
                break

        return self.x

    # first K - 1 Elements
    def __init_working_set(self):
        S = []
        S_a = []
        S_i = []
        v = np.empty((1, self.n), dtype=[('idx', int), ('val', float)])
        for i in range(self.n):
            w_best = self.__find_optimal_gradient_displacement(self.x[i], self.gradient[i])
            v[i] = (i, abs((w_best - self.x[i]) * self.gradient[i]))

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

    def __solve_small_QP(self, S):
        u_k = null_space(self.C[:, S])
        a_k = self.__find_optimal_solution(self.x, self.dF, self.A, self.C, S)
        dx_s = np.zeros((u_k.shape[0], 1))
        for idx, a in enumerate(a_k):
            dx_s += a * u_k[:, idx]
        return dx_s

    def __find_optimal_solution(self, x, dF, A, C, S):
        D = self.K - np.linalg.matrix_rank(C[:, S])
        bounds = []
        for i in range(D):
            bounds.append(self.__get_bounds(i, x, C[:, S]))
        bounds = tuple(bounds)

    def __get_bounds(self, ):
        pass
