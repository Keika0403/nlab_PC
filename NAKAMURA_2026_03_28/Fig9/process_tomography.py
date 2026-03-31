import numpy as np
import cvxpy as cvx
# import libys.qtp as ysq
import matplotlib.pyplot as plt
import itertools
from scipy.linalg import sqrtm

ket_g = np.array([1, 0])
ket_e = np.array([0, 1])
ket_p = np.array([1, 1]) / np.sqrt(2)
ket_i = np.array([1, 1j]) / np.sqrt(2)
ket_m = np.array([1, -1]) / np.sqrt(2)
ket_mi = np.array([1, -1j]) / np.sqrt(2)

g_trit = np.insert(ket_g, 2, 0)
e_trit = np.insert(ket_e, 2, 0)
f_trit = np.insert(ket_e, 0, 0)
p_ge = np.insert(ket_p, 2, 0)
m_ge = np.insert(ket_m, 2, 0)
p_ef = np.insert(ket_p, 0, 0)
p_gf = np.insert(ket_p, 1, 0)
i_ge = np.insert(ket_i, 2, 0)
mi_ge = np.insert(ket_mi, 2, 0)
i_ef = np.insert(ket_i, 0, 0)
i_gf = np.insert(ket_i, 1, 0)

lambda0 = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1]]).reshape(3, 3)
lambda1 = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0]]).reshape(3, 3)* np.sqrt(3/2)
lambda2 = np.array([[0, -1j, 0, 1j, 0, 0, 0, 0, 0]]).reshape(3, 3)* np.sqrt(3/2)
lambda3 = np.array([[1, 0, 0, 0, -1, 0, 0, 0, 0]]).reshape(3, 3)* np.sqrt(3/2)
lambda4 = np.array([[0, 0, 1, 0, 0, 0, 1, 0, 0]]).reshape(3, 3)* np.sqrt(3/2)
lambda5 = np.array([[0, 0, -1j, 0, 0, 0, 1j, 0, 0]]).reshape(3, 3)* np.sqrt(3/2)
lambda6 = np.array([[0, 0, 0, 0, 0, 1, 0, 1, 0]]).reshape(3, 3)* np.sqrt(3/2)
lambda7 = np.array([[0, 0, 0, 0, 0, -1j, 0, 1j, 0]]).reshape(3, 3)* np.sqrt(3/2)
lambda8 = np.array([[1, 0, 0, 0, 1, 0, 0, 0, -2]]).reshape(3, 3) / np.sqrt(3)* np.sqrt(3/2)
lambda_list = np.array([lambda0, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, lambda8,])

def is_unitary(matrix, tol=1e-10):
    """
    行列がユニタリかどうかをチェックする関数。
    
    Args:
    -
    matrix (numpy.ndarray): チェックする行列。
    tol (float): 許容誤差。デフォルトは1e-10。
    
    Returns
    ----
    bool: 行列がユニタリであればTrue、そうでなければFalse。
    """
    # 行列の形状をチェック
    if matrix.shape[0] != matrix.shape[1]:
        return False  # 正方行列でない場合はユニタリでない

    # 行列の随伴行列（共役転置行列）を計算
    matrix_dagger = np.conjugate(matrix.T)

    # 単位行列との違いを計算
    identity_matrix = np.eye(matrix.shape[0])
    difference = np.dot(matrix, matrix_dagger) - identity_matrix

    # 許容誤差以内かどうかをチェック
    return np.allclose(difference, np.zeros_like(matrix), atol=tol)

def Matrixplot(M, figsize=(2,2)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    cax  = ax.imshow(M, cmap="bwr", vmax=1, vmin=-1)
    # 目盛の位置を指定
    xticks = np.arange(M.shape[0])
    yticks = np.arange(M.shape[1])

    l = ["I", "X", "Y", "Z"]
    l_lambda = [f"$\lambda_{i}$" for i in range(9)]
    if M.shape[0] == 4:
        xlabels = ylabels = l
    elif M.shape[0] == 16:
        combinations = list(itertools.product(l, repeat=2))
        xlabels = ylabels = [''.join(combo) for combo in combinations]
    elif M.shape[0] == 9:
        xlabels = ylabels = l_lambda
    elif M.shape[0] == 81:
        combinations = list(itertools.product(l_lambda, repeat=2))
        xlabels = ylabels = [''.join(combo) for combo in combinations]
    else: 
        xlabels = [f"{i}" for i in range(M.shape[0])]
        ylabels = [f"{i}" for i in range(M.shape[0])]

    # 目盛のラベルを指定
    xticklabels = xlabels
    yticklabels = ylabels

    # X軸の目盛とラベルを設定
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    # X軸の目盛とラベルを設定
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    cbar = plt.colorbar(cax)
    cbar.set_label(' ', rotation=270, labelpad=15)
    ax.grid(True, linestyle='--', color='gray', linewidth=0.5)
    plt.show()

def QPT_MaxLik_Qubit(final_states:np.ndarray):
    init_states = (ket_g, ket_p, ket_i, ket_e)
    Erho_g = final_states[0]
    Erho_p = final_states[1]
    Erho_i = final_states[2]
    Erho_e = final_states[3]
    Erho1 = Erho_g
    Erho4 = Erho_e
    Erho2 = Erho_p - 1j*Erho_i - (1-1j) / 2 * (Erho1 + Erho4)
    Erho3 = Erho_p + 1j*Erho_i - (1+1j) / 2 * (Erho1 + Erho4)

    I = np.array([1, 0, 0, 1]).reshape(2, 2)
    sigmax = np.array([0, 1, 1, 0]).reshape(2, 2)

    Lambda = np.block([[I, sigmax], [sigmax, -I]]) / 2
    rho_matrix = np.block([[Erho1, Erho3], [Erho2, Erho4]])
    chi = Lambda @ rho_matrix @ Lambda
    return chi

def QPT_cvx_Qubit(final_states:np.ndarray):
    init_states = np.array([
        np.outer(ket_g, ket_g.conj()),
        np.outer(ket_p, ket_p.conj()),
        np.outer(ket_m, ket_m.conj()),
        np.outer(ket_i, ket_i.conj()),
        np.outer(ket_mi, ket_mi.conj()),
        np.outer(ket_e, ket_e.conj()),
    ], dtype=complex)
    assert len(init_states) == len(final_states)

    I = np.array([[1, 0], [0, 1]])
    sigmax = np.array([[0, 1], [1, 0]])
    sigmay = -1j * np.array([[0, -1j], [1j, 0]])
    sigmaz = np.array([[1, 0], [0, -1]])
    Kraus_list = np.array([I, sigmax, sigmay, sigmaz])

    chi = cvx.Variable((4, 4), hermitian=True)
    cost_func = 0
    for k, init_rho in enumerate(init_states):
        final_rho = final_states[k]
        E_init_rho = 0
        for m in range(chi.shape[0]):
            for n in range(chi.shape[1]):
                E_init_rho += (Kraus_list[m] @ init_rho @ Kraus_list[n]) * chi[m, n]
        cost_func += cvx.sum(cvx.abs(E_init_rho - final_rho)**2)
    
    objective = cvx.Minimize(cost_func)
    constraints = [chi >> 0]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve()
    return chi.value

def QPT_complete(final_states: np.ndarray):
    # input states
    ket_g = np.array([1, 0])  # |g⟩
    ket_e = np.array([0, 1])  # |e⟩
    ket_plus = (ket_g + ket_e)/np.sqrt(2)  # |+⟩ = (|g⟩ + |e⟩)/√2  
    ket_minus = (ket_g - ket_e)/np.sqrt(2)  # |-⟩ = (|g⟩ - |e⟩)/√2
    ket_i = (ket_g + 1j*ket_e)/np.sqrt(2)  # |i⟩ = (|g⟩ + i|e⟩)/√2
    ket_mi = (ket_g - 1j*ket_e)/np.sqrt(2)  # |-i⟩ = (|g⟩ - i|e⟩)/√2
    
    input_states = [ket_g, ket_e, ket_plus, ket_minus, ket_i, ket_mi]
    input_rhos = [np.outer(state, state.conj()) for state in input_states]

    # Pauli basis operators
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    basis = [I, X, Y, Z]

    # β matrix and λ vector
    n_inputs = len(input_rhos)
    A = np.zeros((n_inputs*4, 16), dtype=complex)
    b = np.zeros(n_inputs*4, dtype=complex)

    for i, rho_in in enumerate(input_rhos):
        rho_out = final_states[i]

        # calculate the matrix elements
        for j, E1 in enumerate(basis):
            for k, E2 in enumerate(basis):
                idx = i*4 + j
                col = j*4 + k
                A[idx, col] = np.trace(E1 @ rho_in @ E2.conj().T)

        # vectrize the output states
        b[i*4:(i+1)*4] = rho_out.flatten()

    # Linear inversion
    chi = np.linalg.lstsq(A, b, rcond=None)[0]
    chi = chi.reshape(4, 4)

    return chi

def PTM_single_qubit(final_states, plot=False):
    """
    An array "final_states" corresponds to |g⟩⟨g|, |+⟩⟨+|, |i⟩⟨i|, |e⟩⟨e|
    """
    d = 2
    final_states = np.array(final_states)
    A = np.array([[1, -1, -1, 1],
                    [0, 2, 0, 0],
                    [0, 0, 2, 0],
                    [1, -1, -1, -1]])
    E_sigma_list = np.einsum("ij,ikl->jkl", A, final_states)
    I = np.array([1, 0, 0, 1]).reshape(2, 2)
    sigmax = np.array([0, 1, 1, 0]).reshape(2, 2)
    sigmay = np.array([0, -1j, 1j, 0]).reshape(2, 2)
    sigmaz = np.array([1, 0, 0, -1]).reshape(2, 2)
    sigma_list = [I, sigmax, sigmay, sigmaz]

    R = np.empty((d**2, d**2), dtype=complex)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            R[i, j] = np.trace(sigma_list[i] @ E_sigma_list[j]) / d
    R = R.real
    if plot:
        Matrixplot(R)
    return R

def PTM_two_qubits(final_states, plot=False):
    """
    input_states : (g, p, i, e) \otimes (g, p, i, e) = (gg, gp, gi, ge, pg, ...)
    """
    d = 4 # 2^2
    final_states = np.array(final_states).reshape(d, d, d, d)
    A = np.array([[1, -1, -1, 1],
                    [0, 2, 0, 0],
                    [0, 0, 2, 0],
                    [1, -1, -1, -1]])
    E_on_only_second = np.einsum("ji,kjmn->kimn", A, final_states)
    E_sigma_list = np.einsum("ji,jkmn->ikmn", A, E_on_only_second).reshape(d**2, d, d)
    I = np.array([1, 0, 0, 1]).reshape(2, 2)
    X = np.array([0, 1, 1, 0]).reshape(2, 2)
    Y = np.array([0, -1j, 1j, 0]).reshape(2, 2)
    Z = np.array([1, 0, 0, -1]).reshape(2, 2)
    single_sigma_list = np.array([I, X, Y, Z])
    sigma_list = np.kron(single_sigma_list, single_sigma_list)
    R = np.empty((d**2, d**2), dtype=complex)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            R[i, j] = np.trace(sigma_list[i] @ E_sigma_list[j]) / d
    R = R.real
    if plot:
        Matrixplot(R)
    return R

def PTM_single_qutrit(final_states, plot=False):
    """
    input_states : (g, p_ge, i_ge, e, p_gf, i_gf, f, p_ef, i_ef) ^2
    """
    d = 3
    final_states = np.array(final_states)
    A = np.array([[1, -1, -1, 1, -1, -1, 0, 0, 1/np.sqrt(3)],
              [0, 2, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 2, 0, 0, 0, 0, 0, 0,],
              [1, -1, -1, -1, 0, 0, -1, -1, 1/np.sqrt(3)],
              [0, 0, 0, 0, 2, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 2, 0, 0, 0],
              [1, 0, 0, 0, -1, -1, -1, -1, -2/np.sqrt(3)],
              [0, 0, 0, 0, 0, 0, 2, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 2, 0]])
    A[:, 1:9] *= np.sqrt(3/2)
    E_lambda_list = np.einsum("ij,ikl->jkl", A, final_states)

    R = np.empty((d**2, d**2), dtype=complex)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            R[i, j] = np.trace(lambda_list[i] @ E_lambda_list[j]) / d
    R = R.real
    if plot:Matrixplot(R)
    return R

def PTM_two_qutrits(final_states, plot=False):
    """
    input_states : (g, p_ge, i_ge, e, p_gf, i_gf, f, p_ef, i_ef) ^2
    """
    d = 9
    final_states = np.array(final_states).reshape(d, d, d, d)
    A = np.array([[1, -1, -1, 1, -1, -1, 0, 0, 1/np.sqrt(3)],
              [0, 2, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 2, 0, 0, 0, 0, 0, 0,],
              [1, -1, -1, -1, 0, 0, -1, -1, 1/np.sqrt(3)],
              [0, 0, 0, 0, 2, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 2, 0, 0, 0],
              [1, 0, 0, 0, -1, -1, -1, -1, -2/np.sqrt(3)],
              [0, 0, 0, 0, 0, 0, 2, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 2, 0]])
    A[:, 1:9] *= np.sqrt(3/2)
    E_on_only_second = np.einsum("ji,kjmn->kimn", A, final_states)
    E_sigma_list = np.einsum("ji,jkmn->ikmn", A, E_on_only_second).reshape(d**2, d, d)
    single_sigma_list = np.array([lambda0, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, lambda8, ])
    sigma_list = np.kron(single_sigma_list, single_sigma_list)
    R = np.empty((d**2, d**2), dtype=complex)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            R[i, j] = np.trace(sigma_list[i] @ E_sigma_list[j]) / d
            if R[i, j].imag > 0.1:
                print(f"{i} {j}, {R[i, j]:.3f}")
                print(f"{np.round(sigma_list[i], 3)}")
                print(f"{np.round(E_sigma_list[j], 3)}")
    R = R.real
    if plot:
        Matrixplot(R)
    return R

def fidelity_R_PTM(R:np.ndarray, R_ideal:np.ndarray):
    d = np.sqrt(R.shape[0])
    if d == 2 or d == 4:
        return (np.trace(R_ideal.conj().T @ R).real + d) / (d**2 + d)
    elif d==3 or d==9:
        return np.trace(R_ideal.conj().T @ R) / d**2

def fidelity_chi_PTM(R:np.ndarray, R_ideal:np.ndarray):
    d = np.sqrt(R.shape[0])
    F_R = fidelity_R_PTM(R, R_ideal)
    return ((d + 1) * F_R - 1) / d

