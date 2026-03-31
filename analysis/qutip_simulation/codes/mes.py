import numpy as np
import qutip as qt
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.linalg import eigh
from scipy import interpolate as interp
from mpmath import polylog


class Plot:
    def __init__(self, data, au_input, aud_input, c_res, cd_res) -> None:
        self.data = data
        self.au_input = au_input


    def mesolve_plot(self):
        # Extract states
        states = self.states

        # Input operators
        au_s = qt.expect(au_input, states)
        aud_s = qt.expect(aud_input, states)
        audau_s = qt.expect(aud_input * au_input, states)

        # Resonator operators
        c_s = qt.expect(c_res, states)
        cd_s = qt.expect(cd_res, states)
        cdc_s = qt.expect(cd_res * c_res, states)

        # Qubit operators
        b_s = qt.expect(b_q, states)
        bd_s = qt.expect(bd_q, states)
        bdb_s = qt.expect(bd_q * b_q, states)

        # Cross terms
        audc_s = qt.expect(aud_input * c_res, states)
        cdau_s = qt.expect(cd_res * au_input, states)

        # Lindblad term
        gu_values = np.array([g_u(t, args0) for t in tList])
        LdL = (
            gu_values * np.conjugate(gu_values) * audau_s
            + kappa_ext0 * cdc_s
            + np.sqrt(kappa_ext0) * (gu_values * audc_s + np.conjugate(gu_values) * cdau_s)
        )
        fig, axes = plt.subplots(1, 5, figsize=(20, 6))
        plt.subplots_adjust(wspace=0.4)

        # Input operators
        axes[0].plot(tList, au_s, label=r"$\langle a_u \rangle$")
        axes[0].plot(tList, aud_s, label=r"$\langle a_u^\dagger \rangle$")
        axes[0].plot(tList, audau_s, label=r"$\langle a_u^\dagger a_u \rangle$")
        axes[0].legend()
        axes[0].set_title("Input Operators")
        axes[0].set_xlabel(r"Time (units of $\kappa^{-1}$)")
        axes[0].set_ylabel("Expectation Value")

        # Resonator operators
        axes[1].plot(tList, c_s, label=r"$\langle c \rangle$")
        axes[1].plot(tList, cd_s, label=r"$\langle c^\dagger \rangle$")
        axes[1].plot(tList, cdc_s, label=r"$\langle c^\dagger c \rangle$")
        axes[1].legend()
        axes[1].set_title("Resonator Operators")
        axes[1].set_xlabel(r"Time (units of $\kappa^{-1}$)")
        axes[1].set_ylabel("Expectation Value")

        # Resonator operators
        axes[2].plot(tList, b_s, label=r"$\langle b \rangle$")
        axes[2].plot(tList, bd_s, label=r"$\langle b^\dagger \rangle$")
        axes[2].plot(tList, bdb_s, label=r"$\langle b^\dagger b \rangle$")
        axes[2].legend()
        axes[2].set_title("Resonator Operators")
        axes[2].set_xlabel(r"Time (units of $\kappa^{-1}$)")
        axes[2].set_ylabel("Expectation Value")

        # Comparison between key terms
        axes[3].plot(tList, audau_s, label=r"$\langle a_u^\dagger a_u \rangle$")
        axes[3].plot(tList, cdc_s, label=r"$\langle c^\dagger c \rangle$")
        axes[3].plot(tList, bdb_s, label=r"$\langle b^\dagger b \rangle$")
        axes[3].legend()
        axes[3].set_title("Comparison of Operators")
        axes[3].set_xlabel(r"Time (units of $\kappa^{-1}$)")
        axes[3].set_ylabel("Expectation Value")

        # Cross terms
        axes[4].plot(tList, np.sqrt(LdL), label=r"$\sqrt{<L_dL>}$")
        axes[4].legend()
        axes[4].set_title("Reflected Waveform")
        axes[4].set_xlabel(r"Time (units of $\kappa^{-1}$)")
        axes[4].set_ylabel("Amplitude")

        plt.show()
