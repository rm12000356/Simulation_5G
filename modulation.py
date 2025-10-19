import numpy as np

def nr_modulate(bits, mod_order='QPSK'):
    """
    Modulates bits to QPSK symbols (Gray-coded, unit power).
    bits: np.array 0/1, length multiple of 2.
    """
    if mod_order != 'QPSK':
        raise ValueError("Only QPSK for control sim")
    bits_per_sym = 2
    # Const: idx 0=00 (1+1j), 1=01 (1-1j), 2=10 (-1+1j), 3=11 (-1-1j)
    const = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    num_syms = len(bits) // bits_per_sym
    symbols = np.zeros(num_syms, dtype=complex)
    for i in range(num_syms):
        # Pack: b0 (MSB) *2 + b1 (LSB) → idx
        idx = bits[2*i] * 2 + bits[2*i + 1]
        symbols[i] = const[idx]
    return symbols

def nr_demodulate(rx_symbols, mod_order='QPSK'):
    """
    Hard demod: Min distance → bits (reverses packing).
    rx_symbols: complex array (noisy or clean).
    """
    if mod_order != 'QPSK':
        raise ValueError("Only QPSK for control sim")
    bits_per_sym = 2
    # Same const
    const = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    num_syms = len(rx_symbols)
    rx_bits = np.zeros(num_syms * bits_per_sym, dtype=np.int32)
    for i in range(num_syms):
        # Find closest idx
        dists = np.abs(rx_symbols[i] - const)
        idx = np.argmin(dists)
        # Unpack: b0 = idx // 2 (MSB), b1 = idx % 2 (LSB)
        b0 = idx // 2
        b1 = idx % 2
        rx_bits[2*i] = b0
        rx_bits[2*i + 1] = b1
    return rx_bits