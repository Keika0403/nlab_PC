import numpy as np
import cvxpy as cvx
# from .libys import qtp as ysq
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
p_ef = np.insert(ket_p, 0, 0)
p_gf = np.insert(ket_p, 1, 0)
i_ge = np.insert(ket_i, 2, 0)
i_ef = np.insert(ket_i, 0, 0)
i_gf = np.insert(ket_i, 1, 0)

def QST_fidelity(rho:np.ndarray, rho_target:np.ndarray):
    return np.trace(sqrtm(sqrtm(rho) @ rho_target @ sqrtm(rho)))**2

def QST_MaxLik_Qubit(measured_prob,iteration=10000):
    """
    input measured probability "mesured_prob" should be an array

    POVMs = |g⟩⟨g|, |+⟩⟨+|, |i⟩⟨i|, |e⟩⟨e|
    """
    d = 2
    mesured_prob=np.array(measured_prob)
    POVMarray = np.array([
        np.outer(ket_g, ket_g.conj()),
        np.outer(ket_p, ket_p.conj()),
        np.outer(ket_i, ket_i.conj()),
        np.outer(ket_e, ket_e.conj()),
    ], dtype=complex)
    G = np.linalg.inv(np.sum(POVMarray,axis=0))
    intrho=np.diag(np.ones(d)/d)
    LikelihoodList=[]    
    for ss in range(iteration):
        # 各POVMに対する確率
        Prob = np.real([np.trace(intrho@POVMarray[cc]) for cc in range(len(POVMarray))])
        L = np.prod(Prob**mesured_prob) # 尤度関数
        LikelihoodList.append(L)
        f_over_Prob = np.nan_to_num(mesured_prob/Prob, nan=0.0) # 頻度/確率の配列
        R=np.sum(f_over_Prob[:, np.newaxis, np.newaxis] * POVMarray, axis=0)
        tmprho = G @ R @ intrho @ R @ G
        # tmprho=R@intrho@R
        intrho=tmprho/np.trace(tmprho)
    rho=intrho
    return rho

def QST_cvx_Qubit(measured_prob,iteration=10000):
    """
    input measured probability "mesured_prob" should be an array

    POVMs = |g⟩⟨g|, |+⟩⟨+|, |i⟩⟨i|, |e⟩⟨e|
    """
    d = 2
    mesured_prob=np.array(measured_prob)
    POVMarray = np.array([
        np.outer(ket_g, ket_g.conj()),
        np.outer(ket_p, ket_p.conj()),
        np.outer(ket_i, ket_i.conj()),
        np.outer(ket_e, ket_e.conj()),
    ], dtype=complex)
    rho = cvx.Variable((d, d), hermitian=True)
    cost_func = 0

    for i in range(len(POVMarray)):
        diff_real = cvx.abs(mesured_prob[i]-cvx.real(cvx.trace(rho @ POVMarray[i])))
        cost_func += diff_real

    objective = cvx.Minimize(cost_func)
    constraints = [rho == cvx.Variable((d, d), hermitian=True),                   
                   cvx.trace(rho) == 1,
                   rho >> 0]

    prob = cvx.Problem(objective, constraints)
    result = prob.solve()
    print("obj: ", prob.value)
    return rho.value

def QST_MaxLik_two_Qubits(measured_prob,iteration=10000):
    """
    single_POVMs =  |g⟩⟨g|, 
                    |+⟩⟨+|, 
                    |i⟩⟨i|, 
                    |e⟩⟨e|, 
    """
    d = 2**2
    mesured_prob=np.array(measured_prob)
    assert len(measured_prob) == d**2
    singlePOVMarray = np.array([
        np.outer(ket_g, ket_g.conj()),
        np.outer(ket_p, ket_p.conj()),
        np.outer(ket_i, ket_i.conj()),
        np.outer(ket_e, ket_e.conj()),
    ], dtype=complex)
    POVMarray = np.empty((d, d, d, d,), dtype=complex)
    for i in range(singlePOVMarray.shape[0]):
        for j in range(singlePOVMarray.shape[0]):
            POVMarray[i,j] = np.kron(singlePOVMarray[i], singlePOVMarray[j])
    G = np.linalg.inv(np.sum(POVMarray,axis=(0, 1)))
    intrho=np.diag(np.ones(d)/d)
    LikelihoodList=[]    
    for ss in range(iteration):
        Prob = np.real([np.trace(intrho@singlePOVMarray[cc]) for cc in range(len(singlePOVMarray))]) # 各POVMに対する確率
        L = np.prod(Prob**mesured_prob) # 尤度関数
        LikelihoodList.append(L)
        f_over_Prob = np.nan_to_num(mesured_prob/Prob, nan=0.0) # 頻度/確率の配列
        R=np.sum(f_over_Prob[:, np.newaxis, np.newaxis] * singlePOVMarray, axis=0)
        tmprho = G @ R @ intrho @ R @ G
        # tmprho=R@intrho@R
        intrho=tmprho/np.trace(tmprho)
    rho=intrho
    return rho

def QST_MaxLik_Qutrit(measured_prob,iteration=5000):
    """
    input measured probability "mesured_prob" should be an array

    POVMs = |g⟩⟨g|, |+_ge⟩⟨+_ge|, |i_ge⟩⟨i_ge|, |e⟩⟨e|, |+_ge⟩⟨+_ge|, |i_ge⟩⟨i_ge|, |f⟩⟨f|, |p_ef⟩⟨p_ef|, |i_ef⟩⟨i_ef|
    """
    mesured_prob=np.array(measured_prob)
    POVMarray = np.array([
        np.outer(g_trit, g_trit.conj()),
        np.outer(p_ge, p_ge.conj()),
        np.outer(i_ge, i_ge.conj()),
        np.outer(e_trit, e_trit.conj()),
        np.outer(p_gf, p_gf.conj()),
        np.outer(i_gf, i_gf.conj()),
        np.outer(f_trit, f_trit.conj()),
        np.outer(p_ef, p_ef.conj()),
        np.outer(i_ef, i_ef.conj()),
    ], dtype=complex)
    G = np.linalg.inv(np.sum(POVMarray,axis=0))
    intrho=np.diag(np.ones(3)*1/3)
    LikelihoodList=[]    
    for ss in range(iteration):
        Prob = np.real([np.trace(intrho@POVMarray[cc]) for cc in range(len(POVMarray))]) # 各POVMに対する確率
        L = np.prod(Prob**mesured_prob) # 尤度関数
        LikelihoodList.append(L)
        f_over_Prob = np.nan_to_num(mesured_prob/Prob, nan=0.0) # 頻度/確率の配列
        R=np.sum(f_over_Prob[:, np.newaxis, np.newaxis] * POVMarray, axis=0)
        tmprho = G @ R @ intrho @ R @ G
        # tmprho=R@intrho@R
        intrho=tmprho/np.trace(tmprho)
    rho=intrho
    return rho

def QST_cvx_Qutrit(measured_prob,iteration=5000):
    """
    input measured probability "mesured_prob" should be an array

    POVMs = |g⟩⟨g|, |+_ge⟩⟨+_ge|, |i_ge⟩⟨i_ge|, |e⟩⟨e|, |+_ge⟩⟨+_ge|, |i_ge⟩⟨i_ge|, |f⟩⟨f|, |p_ef⟩⟨p_ef|, |i_ef⟩⟨i_ef|
    """
    d=3
    mesured_prob=np.array(measured_prob)
    POVMarray = np.array([
        np.outer(g_trit, g_trit.conj()),
        np.outer(p_ge, p_ge.conj()),
        np.outer(i_ge, i_ge.conj()),
        np.outer(e_trit, e_trit.conj()),
        np.outer(p_gf, p_gf.conj()),
        np.outer(i_gf, i_gf.conj()),
        np.outer(f_trit, f_trit.conj()),
        np.outer(p_ef, p_ef.conj()),
        np.outer(i_ef, i_ef.conj()),
    ], dtype=complex)
    rho = cvx.Variable((d, d), hermitian=True)
    cost_func = 0

    for i in range(len(POVMarray)):
        diff_real = cvx.abs(mesured_prob[i]-cvx.real(cvx.trace(rho @ POVMarray[i])))
        cost_func += diff_real

    objective = cvx.Minimize(cost_func)
    constraints = [rho == cvx.Variable((d, d), hermitian=True),                   
                   cvx.trace(rho) == 1,
                   rho >> 0]

    prob = cvx.Problem(objective, constraints)
    result = prob.solve()
    print("obj: ", prob.value)
    return rho.value

def QST_MaxLik_two_Qutrits(measured_prob,iteration=10000):
    ## input measured probability "mesured_prob" should be an array
    """
    single_POVMs =  |g⟩⟨g|, 
                    |+_ge⟩⟨+_ge|, 
                    |i_ge⟩⟨i_ge|, 
                    |e⟩⟨e|, 
                    |+_ge⟩⟨+_ge|, 
                    |i_ge⟩⟨i_ge|, 
                    |f⟩⟨f|, 
                    |p_ef⟩⟨p_ef|, 
                    |i_ef⟩⟨i_ef|
    """
    d = 3**2
    mesured_prob=np.array(measured_prob)
    assert len(measured_prob) == d**2
    singlePOVMarray = np.array([
        np.outer(g_trit, g_trit.conj()),
        np.outer(p_ge, p_ge.conj()),
        np.outer(i_ge, i_ge.conj()),
        np.outer(e_trit, e_trit.conj()),
        np.outer(p_gf, p_gf.conj()),
        np.outer(i_gf, i_gf.conj()),
        np.outer(f_trit, f_trit.conj()),
        np.outer(p_ef, p_ef.conj()),
        np.outer(i_ef, i_ef.conj()),
    ], dtype=complex)
    POVMarray = np.empty((d, d, d, d,), dtype=complex)
    for i in range(singlePOVMarray.shape[0]):
        for j in range(singlePOVMarray.shape[0]):
            POVMarray[i,j] = np.kron(singlePOVMarray[i], singlePOVMarray[j])
    G = np.linalg.inv(np.sum(POVMarray,axis=(0, 1)))
    intrho=np.diag(np.ones(d)/d)
    LikelihoodList=[]    
    for ss in range(iteration):
        Prob = np.real([np.trace(intrho@singlePOVMarray[cc]) for cc in range(len(singlePOVMarray))]) # 各POVMに対する確率
        L = np.prod(Prob**mesured_prob) # 尤度関数
        LikelihoodList.append(L)
        f_over_Prob = np.nan_to_num(mesured_prob/Prob, nan=0.0) # 頻度/確率の配列
        R=np.sum(f_over_Prob[:, np.newaxis, np.newaxis] * singlePOVMarray, axis=0)
        tmprho = G @ R @ intrho @ R @ G
        # tmprho=R@intrho@R
        intrho=tmprho/np.trace(tmprho)
    rho=intrho
    return rho
