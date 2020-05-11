import numpy as np
from scipy.linalg import null_space
from scipy.optimize import lsq_linear, minimize


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
            j_all = [i for i in range(self.n)]
            j_without_S = np.setdiff1d(j_all, S)
            for j in j_without_S:
                S.append(j)
                dx_S_best = self.__solve_small_QP(S)
                dF_temp = abs(
                    dx_S_best.transpose().dot(self.A[:, S].transpose()[:, S].transpose()).dot(
                        dx_S_best) + self.gradient[S].transpose().dot(
                        dx_S_best))
                if dF_temp > dF_best:
                    dF_best = dF_temp
                    j_best = j
                    dx_best_S_best = dx_S_best
                S.remove(j)

            if j_best == -1:
                raise RuntimeError("cant find second best choice to optimise")

            S.append(j_best)
            self.x[S] += dx_best_S_best
            self.gradient += self.step_size * (self.A + self.A.transpose() + np.diag(self.b))[:, S].dot(dx_best_S_best)

            if dF_best < self.epsilon:
                print("Delta F < EPSILON")
                print(t)
                print(dF_best)
                return self.x

        print("Max Iter reached")
        return self.x

    # first K - 1 Elements
    def __init_working_set(self):
        S = []
        S_a = []
        S_i = []
        v = np.empty((self.n,), dtype=[('idx', int), ('val', float)])
        for i in range(self.n):
            w_best = self.__find_optimal_gradient_displacement(self.x[i], self.gradient[i])
            v[i] = (i, abs((w_best - self.x[i]) * self.gradient[i]))

            if not v[i] == 0:
                S_a.append(i)
            else:
                S_i.append(i)
        if len(S_a) > self.K:
            p_upperbound = round(len(S_a) * 0.1)
            p = np.random.choice([i for i in range(p_upperbound)], 1)[0]

            sorted_v = np.sort(v, order='val')
            for i in range(self.K - p - 1):
                S.append(sorted_v[i][0])

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
        a_k = self.__find_optimal_solution(S)
        dx_s = np.zeros((u_k.shape[0], 1))
        for idx, a in np.ndenumerate(a_k):
            dx_s += a * u_k[:, idx]
        return dx_s.reshape((dx_s.shape[0],))

    def __find_optimal_solution(self, S):
        D = np.linalg.matrix_rank(self.C) - np.linalg.matrix_rank(self.C[:, S]) + 1
        bounds = []
        for i in range(D):
            bounds.append(self.__get_bounds(self.x[i]))
        bounds = tuple(bounds)

        solution = minimize(fun=objective_function, x0=np.array([1] * D), args=(D, S, self.A, self.gradient),
                            bounds=bounds)
        return solution.x

    def __get_bounds(self, x_i):
        a_min = self.r - x_i
        a_max = self.R - x_i
        return a_min, a_max


def objective_function(a, D, S, A, grad):
    sum1 = 0
    for k in range(D):
        sum1 += (a[k] * a[k]) * A[S[k], S[k]]

    sum2 = 0
    for k in range(D):
        for i in range(D):
            if not i == k:
                sum2 += (a[k] * a[i]) * A[S[i], S[k]]

    sum3 = 0
    for k in range(D):
        sum3 += a[k] * grad[S[k]]

    return sum1 + sum2 + sum3
