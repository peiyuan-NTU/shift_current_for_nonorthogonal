import numpy as np
def get_shift_cond_k(tm, alpha, beta, gamma, omega_s, mu, k, epsilon=0.1):
    sigma_s = np.zeros(len(omega_s))
    get_shift_cond_k_inplace(sigma_s, tm, alpha, beta, gamma, omega_s, mu, k, epsilon=epsilon)
    return sigma_s


def get_shift_cond_k_inplace(sigma_s, tm, alpha, beta, gamma, omega_s, mu, k, epsilon=0.1):
    n_omega_s = len(omega_s)
    Es = get_eig(tm, k).values
    A_beta = get_A(tm, beta, k)
    A_gamma = get_A(tm, gamma, k)
    gdr_beta_alpha = get_generalized_dr(tm, beta, alpha, k)
    gdr_gamma_alpha = get_generalized_dr(tm, gamma, alpha, k)
    constant = -3.0828677458430857 * np.sqrt(1 / np.pi) / epsilon

    for n in range(tm.norbits):
        for m in range(tm.norbits):
            En = Es[n]
            Em = Es[m]
            fn = 1 if En < mu else 0
            fm = 1 if Em < mu else 0
            if fn != fm:
                tmp = constant * (fn - fm) * np.imag(A_beta[m, n] * gdr_gamma_alpha[n, m] + A_gamma[m, n] * gdr_beta_alpha[n, m])
                for i_omega in range(n_omega_s):
                    omega = omega_s[i_omega]
                    sigma_s[i_omega] += tmp * np.exp(-(En - Em - omega)**2 / epsilon**2)

    return None
