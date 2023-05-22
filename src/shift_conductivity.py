from topology import Berry_connection, get_D, get_dHbar, get_dSbar, get_Awbar, get_dAwbar, get_dEs
from src.Basic_tool import get_order, get_two_order, get_eigen_for_tbm
from src.Basic_tool import DEGEN_THRESH
import numpy as np
from src.mesh import create_uniform_mesh

# sigma_s = None

results = np.zeros(3, dtype=np.float64)


def get_dr(tm, alpha, beta, k):
    order_alpha = get_order(alpha)
    order_beta = get_order(beta)
    order_alphabeta = get_two_order(alpha, beta)
    dH_alpha_bar = get_dHbar(tm, order_alpha, k)
    dH_alphabeta_bar = get_dHbar(tm, order_alphabeta, k)
    dS_alpha_bar = get_dSbar(tm, order_alpha, k)
    dS_alphabeta_bar = get_dSbar(tm, order_alphabeta, k)
    Aw_alpha_bar = get_Awbar(tm, alpha, k)
    dAw_alphabeta_bar = get_dAwbar(tm, alpha, order_beta, k)
    Es = get_eigen_for_tbm(tm, k)[0]
    dEs = get_dEs(tm, beta, k)
    D_alpha = get_D(tm, alpha, k)
    D_beta = get_D(tm, beta, k)
    dr = np.zeros((tm.norbits, tm.norbits), dtype=np.complex128)
    tmpH = dH_alpha_bar @ D_beta + dH_alphabeta_bar + D_beta.conj().T @ dH_alpha_bar
    tmpS = dS_alpha_bar @ D_beta + dS_alphabeta_bar + D_beta.conj().T @ dS_alpha_bar
    tmpA = Aw_alpha_bar @ D_beta + dAw_alphabeta_bar + D_beta.conj().T @ Aw_alpha_bar

    for m in range(tm.norbits):
        for n in range(tm.norbits):
            En = Es[n]
            Em = Es[m]
            dEn = dEs[n]
            dEm = dEs[m]

            if abs(En - Em) > DEGEN_THRESH[0]:
                dr[n, m] += 1j * tmpH[n, m] / (Em - En)
                dr[n, m] -= 1j * dEm * dS_alpha_bar[n, m] / (Em - En)
                dr[n, m] -= 1j * Em * tmpS[n, m] / (Em - En)
                dr[n, m] -= 1j * (dEm - dEn) * D_alpha[n, m] / (Em - En)
                dr[n, m] += tmpA[n, m]

    return dr


def get_generalized_dr(tm, alpha, beta, k):
    return get_dr(tm, alpha, beta, k)


def get_shift_cond_k_inplace(tm, alpha, beta, gamma, omega_s, mu, k, epsilon=0.1):
    print("alpha = ", alpha, "beta = ", beta, "gamma = ", gamma, "k = ", k, "mu = ", mu, "epsilon = ", epsilon,
          "omega_s = ", omega_s)
    sigma_s = np.zeros(len(omega_s))
    n_omega_s = len(omega_s)
    Es = get_eigen_for_tbm(tm, k)[0]
    Vs = get_eigen_for_tbm(tm, k)[1]
    # print("Es = ", Es)
    # print("Vs = ", Vs)

    A_beta = Berry_connection(tm=tm,
                              alpha=beta,
                              k=k)
    A_gamma = Berry_connection(tm=tm,
                               alpha=gamma,
                               k=k)
    print("A_beta = ", A_beta, "\n A_gamma = ", A_gamma)
    gdr_beta_alpha = get_generalized_dr(tm, beta, alpha, k)
    # print("gdr_beta_alpha = ", gdr_beta_alpha)
    gdr_gamma_alpha = get_generalized_dr(tm, gamma, alpha, k)
    # print("gdr_gamma_alpha = ", gdr_gamma_alpha)
    constant = -3.0828677458430857 * np.sqrt(1 / np.pi) / epsilon

    for n in range(tm.norbits):
        for m in range(tm.norbits):
            En = Es[n]
            Em = Es[m]
            fn = 1 if En < mu else 0
            fm = 1 if Em < mu else 0
            if fn != fm:
                tmp = constant * (fn - fm) * np.imag(
                    A_beta[m, n] * gdr_gamma_alpha[n, m] + A_gamma[m, n] * gdr_beta_alpha[n, m])
                for i_omega in range(n_omega_s):
                    omega = omega_s[i_omega]
                    # print("i_omega = ", i_omega, "omega = ", omega, "tmp = ", tmp, "En = ", En, "Em = ", Em, "tmp....", tmp * np.exp(-(En - Em - omega) ** 2 / epsilon ** 2))
                    sigma_s[i_omega] += tmp * np.exp(-(En - Em - omega) ** 2 / epsilon ** 2)
    print("\n")
    return sigma_s


def get_shift_cond_k(tm, alpha, beta, gamma, omega_s, mu, k, epsilon=0.1):
    sigma_s = get_shift_cond_k_inplace(tm=tm,
                                       alpha=alpha,
                                       beta=beta,
                                       gamma=gamma,
                                       omega_s=omega_s,
                                       mu=mu,
                                       k=k,
                                       epsilon=epsilon)
    return sigma_s


def get_shift_cond_inner(tm, alpha, beta, gamma, omega_s, mu, mesh_size, epsilon: float = 0.1, batchsize: int = 1):
    def collect_result(result):
        global results
        results += result

    nks = np.prod(mesh_size)
    n_omega_s = len(omega_s)
    sigma_s = np.zeros(n_omega_s)
    all_mesh = list(create_uniform_mesh(mesh_size))
    for k in create_uniform_mesh(mesh_size):
        sigma_s += get_shift_cond_k(tm=tm,
                                    alpha=alpha,
                                    beta=beta,
                                    gamma=gamma,
                                    omega_s=omega_s,
                                    mu=mu,
                                    k=k,
                                    epsilon=epsilon)
    # pool = mp.Pool(20)

    # for k in all_mesh:
    #     pool.apply_async(get_shift_cond_k, args=(tm, alpha, beta, gamma, omega_s, mu, k, epsilon),
    #                      callback=collect_result)

    # sigmas = Parallel(n_jobs=-1)(
    #     delayed(get_shift_cond_k)(tm, alpha, beta, gamma, omegas, mu, k, epsilon=epsilon) for k in
    #     create_uniform_mesh(mesh_size=meshsize))
    brillouin_zone_volume = abs(np.linalg.det(tm.rlat))
    # print("brillouin_zone_volume = ", brillouin_zone_volume)
    # print("nks = ", nks)
    print("sigma_s = ", sigma_s)
    return sigma_s * brillouin_zone_volume / nks


def get_shift_cond(tm, alpha, beta, omega_s, mu, mesh_size, epsilon=0.1, batch_size=1):
    # print("meshsize", meshsize)

    return get_shift_cond_inner(tm=tm,
                                alpha=alpha,
                                beta=beta,
                                gamma=beta,
                                omega_s=omega_s,
                                mu=mu,
                                mesh_size=mesh_size,
                                epsilon=epsilon,
                                batchsize=batch_size)
