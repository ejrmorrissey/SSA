import numpy as np
import itertools as it
from concurrent.futures import ProcessPoolExecutor, wait, as_completed

class SSA(object):
    t_matrix = None
    eigs = None
    pcs = None
    rcs = None

    def __init__(self, data, m):
        self.data = np.array(data)
        self.m = m
        self.n = len(self.data)
        self.data = self.data.reshape((self.n, 1))
        self._compute_t_matrix()
        self._compute_eigs()
        self._compute_pcs()

    def _compute_t_matrix(self):
        n = self.n
        m = self.m
        data = self.data
        covariances = [(1 / (n - j)) * np.matmul(data.T[0, 0:n - j - 1], data[j:n - 1, 0]) for j in range(m)]
        toeplitz_matrix = np.array([covariances[abs(i - j)]
                                    for i, j in it.product(*([range(m)] * 2))]).reshape((m, m))
        self.t_matrix = toeplitz_matrix

    def _compute_eigs(self):
        eig_values, eig_vectors = np.linalg.eigh(self.t_matrix)
        eigs = [{"value": value, "vector": vector} for value, vector in
                zip(eig_values, list(eig_vectors.T))]
        eigs_sorted = sorted(eigs, key=lambda e: e["value"],
                             reverse=True)  # Sort eigenvalues and eigenvectors by descending eigenvalue
        self.eigs = eigs_sorted

    def _compute_pcs(self):
        k = 0
        n = self.n
        m = self.m
        eigs = self.eigs
        pcs = np.vstack([[np.matmul(self.data[i:i + m, 0], eigs[k]["vector"]) for i in range(n - m + 1)]
                         for k in range(len(eigs))])
        self.pcs = pcs

    def get_rcs(self, subset):
        m = self.m
        n = self.n
        pcs = self.pcs
        eigs = self.eigs
        rcs = []
        for i in range(1, n + 1):
            s = 0
            rc = (1 / len(self._get_pc_window(i))) * sum(
                [np.matmul(pcs[s][self._get_pc_window(i)].T, eigs[s]["vector"][self._get_eig_window(i)])
                 for s in subset])
            rcs.append(rc)
        return rcs

    def _get_pc_window(self, i):
        m = self.m
        n = self.n
        if 1 <= i <= m - 1:
            return range(i - 1, -1, -1)  # close to beginning: from (i - 1)th pc to zeroth pc (descending)
        elif n - m + 2 <= i <= n:
            return range(n - m, i - m - 1, -1)  # close to end: from (n - m)th pc to (i - m)th pc (descending).
        else:
            return range(i - 1, i - m - 1, -1) # 

    def _get_eig_window(self, i):
        m = self.m
        n = self.n
        if 0 <= i <= m - 1:
            return range(0, i)  # close to beginning of series
        elif n - m + 2 <= i <= n:
            return range(i - n + m - 1, m)  # close to end of series
        else:
            return range(0, m)

    def find_statistical_order(self):
        eigs = self.eigs
        m = self.m
        n = self.n
        
        # Generate 100 white noise processes
        with ProcessPoolExecutor() as executor:
            v_futures = [executor.submit(self._generate_white_noise, i) for i in range(100)]
        v = [f.result() for f in wait(v_futures).done]
        
        for p in range(0, len(eigs) - 1):
            # For a given order p, we reconstruct with subset of eigenvectors p+1... m
            subset = list(range(p + 1, m))
            noise_intervals = self._get_noise_intervals(v, subset)
            rcs = self.get_rcs(subset)
            c_ps = [(1 / (n - j)) * np.matmul(rcs[0:n - j - 1],
                                              rcs[j:n - 1]) for j in range(m)]
            beta_intervals = [self.get_beta_interval(c_p, noise_interval) for c_p, noise_interval in zip(c_ps, noise_intervals)]
            delta = np.amax([beta_interval[0] for beta_interval in beta_intervals], axis=0)
            gamma = np.amin([beta_interval[1] for beta_interval in beta_intervals], axis=0)
            if delta <= gamma:
                return p, (delta, gamma)
        return -1, None

    def _get_noise_intervals(self, v, subset):
        m = self.m
        n = self.n
        c_ws = []
        with ProcessPoolExecutor() as executor:
            c_ws = list(executor.map(self._get_noise_correlations, v, [subset]*len(v)))
        means, sems = np.mean(c_ws, axis=0), np.var(c_ws, axis=0)
        noise_intervals = [(mean-1.96*(var**0.5), mean+1.96*(var**0.5)) for mean, var in zip(means, sems)]
        return noise_intervals

    def _get_noise_correlations(self, w_ssa, subset):
        m = self.m
        n = self.n
        rc_w = w_ssa.get_rcs(subset)
        c_w = [(1 / (n - j)) * np.matmul(rc_w[0:n - j - 1],
                                         rc_w[j:n - 1]) for j in range(m)]
        return c_w

    def get_beta_interval(self, c_p, noise_interval):
        # If lower bound is negative and upper bound is positive, there's no upper
        # limit on beta
        if noise_interval[0] < 0 < noise_interval[1]:
            return max(c_p / noise_interval[0], c_p / noise_interval[1]), np.inf

        # If both lower bound and upper bound are positive: beta must be small enough
        # that c_p is greater than lower bound, large enough that c_p is less than
        # upper bound
        elif 0 < noise_interval[0] < noise_interval[1]:
            return c_p / noise_interval[1], c_p / noise_interval[0]

        # If both lower bound and upper bound are negative, beta must be large enough
        # that c_p is greater than lower bound, small enough that c_p is less than
        # upper bound
        elif noise_interval[0] < noise_interval[1] < 0:
            return c_p / noise_interval[0], c_p / noise_interval[1]

        elif noise_interval[0] == 0 or noise_interval[1] == 0:
            # if lower bound negative, beta must be large enough that c_p greater
            # than negative lower bound
            if noise_interval[0] < 0:
                return c_p / noise_interval[0], np.inf
            # if upper bound positive, beta must be large enough that c_p less than
            # upper bound
            elif noise_interval[1] > 0:
                return c_p / noise_interval[1], np.inf

        else:
            return np.nan, np.nan

    def _generate_white_noise(self, seed):
        np.random.seed(seed)
        w_ssa = SSA(data=np.random.normal(0, 1, self.n), m=self.m)
        return w_ssa
