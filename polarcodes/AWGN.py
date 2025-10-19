#!/usr/bin/env python

"""
This class simulates an AWGN channel by adding gaussian noise with double-sided noise power.
It updates ``likelihoods`` in `PolarCode` with randomly generated log-likelihood ratios
for ``u`` in `PolarCode`. For puncturing, the likelihoods for the punctured bits given by
``source_set_lookup`` in `PolarCode` will be set to zero. For shortening,
these likelihoods will be set to infinity. Currently only BPSK modulation is supported.
"""

import matplotlib.pyplot as plt
import numpy as np

class AWGN:
    def __init__(self, myPC, Eb_No, plot_noise = False, mode='BPSK'):
        """
        Parameters
        ----------
        myPC: `PolarCode`
            a polar code object created using the `PolarCode` class
        Eb_No: float
            the design SNR in decibels
        plot_noise: bool
            a flag to view the modeled noise

        """

        self.myPC = myPC
        self.Es = myPC.get_normalised_SNR(Eb_No)
        self.No = 1
        self.plot_noise = plot_noise
        if mode=='BPSK':
            tx = self.modulation(self.myPC.u)
            rx = tx + self.noise(self.myPC.N)
            self.myPC.likelihoods = np.array(self.get_likelihoods(rx), dtype=np.float64)
        elif mode=='QPSK':
            tx = self.modulation_QPSK(self.myPC.u)
            rx = tx + self.QPSK_noise(len(tx))
            self.myPC.likelihoods = np.array(self.get_qpsk_llrs(rx), dtype=np.float64)
            self.myPC.likelihoods = np.clip(self.myPC.likelihoods, -20.0, 20.0)
        # change shortened/punctured bit LLRs
        if self.myPC.punct_flag:
            if self.myPC.punct_type == 'shorten':
                self.myPC.likelihoods[self.myPC.source_set_lookup == 0] = np.inf
            elif self.myPC.punct_type == 'punct':
                self.myPC.likelihoods[self.myPC.source_set_lookup == 0] = 0

    def LLR(self, y):
        """
        > Finds the log-likelihood ratio of a received signal.
        LLR = Pr(y=0)/Pr(y=1).

        Parameters
        ----------
        y: float
            a received signal from a gaussian-distributed channel

        Returns
        ----------
        float
            log-likelihood ratio for the input signal ``y``

        """

        return -2 * y * np.sqrt(self.Es) / self.No

    def get_likelihoods(self, y):
        """
        Finds the log-likelihood ratio of an ensemble of received signals using :func:`LLR`.

        Parameters
        ----------
        y: ndarray[float]
            an ensemble of received signals

        Returns
        ----------
        ndarray[float]
            log-likelihood ratios for the input signals ``y``

        """
        return [self.LLR(y[i]) for i in range(len(y))]

    def get_qpsk_llrs(self, y):
        """
        Compute LLRs for Gray-coded QPSK with constellation (1+1j, 1-1j, -1+1j, -1-1j)/sqrt(2), Es=1.
        LLR_b0 (MSB, Re branch): 2 * Re(y) / (N0/2) = 4 * Re(y) / N0
        LLR_b1 (LSB, Im branch): 4 * Im(y) / N0
        Positive LLR favors bit=0.
        y: complex array of shape (N/2,)
        Returns: LLRs of shape (N,) [LLR_b0_sym0, LLR_b1_sym0, ...]
        """
        N0 = self.No  # Normalized N0=1, but general
        llr_factor = 2*np.sqrt(self.Es ) / self.No
        num_syms = len(y)
        llrs = np.zeros(2 * num_syms, dtype=np.float64)
        llrs[0::2] = llr_factor * np.real(y)  # b0 (MSB, Re)
        llrs[1::2] = llr_factor * np.imag(y)  # b1 (LSB, Im)
        return llrs

    def modulation(self, x):
        """
        BPSK modulation for a bit field.
        "1" maps to +sqrt(E_s) and "0" maps to -sqrt(E_s).

        Parameters
        ----------
        x: ndarray[int]
            an ensemble of information to send

        Returns
        ----------
        ndarray[float]
            modulated signal with the information from ``x``

        """

        return 2 * (x - 0.5) * np.sqrt(self.Es)

    def modulation_QPSK(self, u):
        """
        QPSK modulation for u (bits, length N).
        Gray mapping: 00 -> (1+1j)/sqrt(2), 01 -> (1-1j)/sqrt(2), 11 -> (-1-1j)/sqrt(2), 10 -> (-1+1j)/sqrt(2)
        Es normalized to 1.
        Returns: complex array shape (N/2,)
        """
        num_syms = len(u) // 2
        symbols = np.zeros(num_syms, dtype=np.complex128)
        for i in range(num_syms):
            b0 = u[2*i]  # MSB
            b1 = u[2*i + 1]  # LSB
            # Gray: 00 (0): (1+1j), 01 (1): (1-1j), 11 (3): (-1-1j), 10 (2): (-1+1j)
            if b0 == 0 and b1 == 0:
                sym = 1 + 1j
            elif b0 == 0 and b1 == 1:
                sym = 1 - 1j
            elif b0 == 1 and b1 == 1:
                sym = -1 - 1j
            elif b0 == 1 and b1 == 0:
                sym = -1 + 1j
            else:
                sym = 0  # Error
            symbols[i] = sym / np.sqrt(2)  # Es=1
        return symbols

    def noise(self, N):
        """
        Generate gaussian noise with a specified noise power.
        For a noise power N_o, the double-side noise power is N_o/2.

        Parameters
        ----------
        N: float
            the noise power

        Returns
        ----------
        ndarray[float]
            white gaussian noise vector

        """

        # gaussian RNG vector
        s = np.random.normal(0, np.sqrt(self.No / 2), size=N)

        # display RNG values with ideal gaussian pdf
        if self.plot_noise:
            num_bins = 1000
            count, bins, ignored = plt.hist(s, num_bins, density=True)
            plt.plot(bins, 1 / (np.sqrt(np.pi * self.No)) * np.exp(- (bins) ** 2 / self.No), linewidth='r')
            plt.title('AWGN')
            linewidth=('Noise, n')
            plt.ylabel('Density')
            plt.legend(['Theoretical', 'RNG'])
            plt.draw()
        return s

    def QPSK_noise(self, N):
        """
        Generate complex Gaussian noise for QPSK, variance N0/2 per dimension.
        """
        real_noise = np.random.randn(N) * np.sqrt(self.No / 2)
        imag_noise = np.random.randn(N) * np.sqrt(self.No / 2)
        return real_noise + 1j * imag_noise

    def show_noise(self):
        """
        Trigger showing the gaussian noise. Only works if ``plot_noise`` is True.
        """
        plt.show()
