import numpy as np

class QPSK_AWGN:
    def __init__(self, myPC, rx_symbols, snr_db):
        """
        rx_symbols: complex modulated symbols (assumed unit average symbol energy Es=1)
        snr_db: Es/N0 in dB (common convention). This class will compute N0 and sigma2 accordingly.
        Sets myPC.likelihoods to an LLR array (len = 2*len(rx_symbols)).
        """
        # store
        self.myPC = myPC
        self.rx_symbols = np.array(rx_symbols, dtype=np.complex128)
        # Interpret snr_db as Es/N0 in dB
        self.snr_db = float(snr_db)
        # symbol energy Es (assume modulator normalized to Es = 1)
        self.Es = np.mean(np.abs(self.rx_symbols)**2)
        if self.Es == 0:
            self.Es = 1.0

        # Compute noise variance per complex sample:
        # SNR_linear = Es / N0  --> N0 = Es / SNR_linear
        SNR_linear = 10.0 ** (self.snr_db / 10.0)
        N0 = self.Es / SNR_linear   # noise spectral density
        sigma2 = N0 / 2.0           # variance per real dimension
        sigma = np.sqrt(sigma2)

        
        # NOTE: If rx_symbols already contain noise (e.g., from channel sim), skip adding more noise.
        # For your pipeline you likely passed clean modulated symbols; we'll add noise here.
        
        noise = (np.random.randn(len(self.rx_symbols)) + 1j * np.random.randn(len(self.rx_symbols))) * sigma
        self.rx_noisy = self.rx_symbols #+ noise

        # compute LLRs for Gray QPSK, mapping as in your nr_modulate:
        self.myPC.likelihoods = self.get_qpsk_llrs(self.rx_noisy, sigma2)

    def get_qpsk_llrs(self, y, sigma2):
        """
        Compute LLRs for Gray QPSK where bit0 = MSB -> Re, bit1 = LSB -> Im.
        LLR = 2 * sqrt(Es) * Re(y) / sigma2  if symbols scaled with Es.
        If constellation normalized to Es=1, then:
            LLR_re = 2 * Re(y) / sigma2
        Return array [b0_sym0, b1_sym0, b0_sym1, b1_sym1, ...]
        Positive LLR favors bit=0 per your decoder convention.
        """
        # If constellation has Es != 1, include sqrt(Es) factor:
        Es = self.Es
        # Standard formula:
        snr_linear = 10 ** (self.Es / 10)
        sigma_sq = 1 / (2 * snr_linear)
        llr_factor = 2 / sigma_sq

        # error up here ^^^^^^^
        num_syms = len(y)
        llrs = np.zeros(2 * num_syms, dtype=np.float64)
        llrs[0::2] = llr_factor * np.real(y)   # b0 per symbol
        llrs[1::2] = llr_factor * np.imag(y)   # b1 per symbol
        return llrs
