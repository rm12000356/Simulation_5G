from scipy.fft import ifft, fft
import numpy as np


N_fft = 512
num_subcarriers = 12  # 1 RB
cp_lengths = [32, 16]  # First/last symbol longer (normal CP approx)


# OFDM TX
def ofdm_tx(symbols):
    num_symbols = len(symbols) // num_subcarriers + 1
    # Pad symbols to num_symbols * num_subcarriers
    padded_symbols = np.pad(symbols, (0, num_symbols * num_subcarriers - len(symbols)))
    freq_domain = np.zeros((num_symbols, N_fft), dtype=complex)
    for sym_idx in range(num_symbols):
        start = sym_idx * num_subcarriers
        freq_domain[sym_idx, N_fft//2 - num_subcarriers//2 : N_fft//2 + num_subcarriers//2] = padded_symbols[start:start + num_subcarriers]  # DC-centered
    time_domain = ifft(freq_domain, axis=1) * np.sqrt(N_fft)  # Normalize
    # Add CP per symbol
    tx_signal = np.zeros(num_symbols, dtype=object)
    for sym_idx in range(num_symbols):
        cp_len = cp_lengths[sym_idx % len(cp_lengths)]  # Cycle
        cp = time_domain[sym_idx, -cp_len:]
        tx_signal[sym_idx] = np.concatenate((cp, time_domain[sym_idx]))
    return np.concatenate(tx_signal)  # Flatten time signal

def multipath_channel(tx_signal, cir=[1.0], delays=[0]):
    """Convolve with CIR for ISI."""
    max_delay = max(delays)
    impulse = np.zeros(max_delay + 1, dtype=complex)  # FIXED: dtype=complex
    for d, tap in zip(delays, cir):
        impulse[d] = tap  # Complex OK now
    rx_signal = np.convolve(tx_signal, impulse, mode='same')
    return rx_signal

# OFDM RX (FIXED: epsilon for /0, clip for NaN)
def ofdm_rx(rx_signal, cir_freq=None):
    """CP remove, FFT, equalize."""
    symbol_len = N_fft + cp_lengths[0]
    num_symbols = len(rx_signal) // symbol_len
    time_domain = np.zeros((num_symbols, N_fft), dtype=complex)
    start = 0
    for sym_idx in range(num_symbols):
        cp_len = cp_lengths[sym_idx % len(cp_lengths)]
        symbol_start = start + cp_len
        symbol_end = symbol_start + N_fft
        time_domain[sym_idx] = rx_signal[symbol_start:symbol_end]
        start = symbol_end
    freq_domain = fft(time_domain, axis=1) / np.sqrt(N_fft)
    if cir_freq is None:
        cir_freq = np.ones(N_fft)
    else:
        cir_freq = np.pad(cir_freq, (0, max(0, N_fft - len(cir_freq))))[:N_fft]
    sub_start = N_fft//2 - num_subcarriers//2
    sub_end = N_fft//2 + num_subcarriers//2
    sub_start = max(0, sub_start)
    sub_end = min(N_fft, sub_end)
    cir_freq_slice = cir_freq[sub_start:sub_end]
    if len(cir_freq_slice) == 0:
        cir_freq_slice = np.ones(num_subcarriers)
    cir_freq_replicated = np.tile(cir_freq_slice, (num_symbols, 1))
    # FIXED: Epsilon for /0, clip for NaN
    cir_freq_replicated = np.clip(cir_freq_replicated, 1e-10, np.inf)
    eq_symbols = freq_domain[:, sub_start:sub_end] / cir_freq_replicated
    eq_symbols = np.clip(eq_symbols, -10, 10)  # FIXED: Clip symbols to avoid NaN in LLRs
    return eq_symbols.flatten()