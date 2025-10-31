import numpy as np
import matplotlib.pyplot as plt

def _bits_to_int(bits):
    """bits: 1D array-like of 0/1, MSB first -> integer"""
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val

def _binary_to_gray(x):
    """binary integer -> Gray code integer"""
    return x ^ (x >> 1)

def _bits_group_to_level(bits_group):
    """
    Map n bits (MSB first) to an amplitude level using Gray mapping.
    Levels are odd integers symmetric around zero: -(2^n-1), ... , (2^n-1) step 2.
    """
    n = len(bits_group)
    idx = _bits_to_int(bits_group)            # binary index 0..2^n-1
    gray = _binary_to_gray(idx)              # map to Gray index
    max_idx = (1 << n) - 1                   # 2^n - 1
    level = 2 * gray - max_idx               # maps 0..max_idx -> -(max_idx) .. +(max_idx) with step 2
    return level

def gray_to_binary(gray):
    """Gray code integer -> binary integer"""
    binary = gray
    mask = gray >> 1
    while mask:
        binary ^= mask
        mask >>= 1
    return binary

def level_to_bits(level, M_axis):
    """
    Inverse of _bits_group_to_level for a single axis.
    level: I or Q level
    M_axis: number of levels along the axis (sqrt(M))
    Returns: list of bits for that axis
    """
    # Determine index in levels
    levels = np.arange(-M_axis + 1, M_axis, 2)
    idx = np.argmin(np.abs(levels - level))
    # Convert Gray index back to binary index, then to bits
    binary_idx = gray_to_binary(idx)
    bits_per_axis = int(np.log2(M_axis))
    return [int(b) for b in format(binary_idx, f'0{bits_per_axis}b')]

def modulation_MQAM(u, M):
    """
    Generic square M-QAM modulator with Gray mapping on each axis.
    - u: 1D array-like bits (0/1), length must be multiple of log2(M)
    - M: 16, 64, 256, etc. Must be a perfect square.
    Returns: complex numpy array of symbols, Es normalized to 1.
    """
    u = np.asarray(u).astype(int)
    k = int(np.log2(M))              
    assert (M & (M - 1))==0, "M must be power of two"
    assert M == int(np.sqrt(M))**2 , "M should be a perfect square"
    assert len(u) % k == 0, "bit length must be multiple of log2(M)"

    bits_per_axis = k // 2
    num_syms = len(u) // k
    symbols = np.zeros(num_syms, dtype=np.complex128)

    # Average symbol energy for square QAM: Es = (2/3)*(M-1)
    Es = (2.0/3.0) * (M - 1)
    norm = np.sqrt(Es)

    for i in range(num_syms):
        block = u[i*k:(i+1)*k]
        i_bits = block[:bits_per_axis]   # MSB..LSB for I
        q_bits = block[bits_per_axis:]   # MSB..LSB for Q

        I = _bits_group_to_level(i_bits)
        Q = _bits_group_to_level(q_bits)

        symbols[i] = (I + 1j * Q) / norm

    return symbols

def demodulate_MQAM(symbols, M):
    k = int(np.log2(M))
    
    m_side = int(np.sqrt(M))
    norm = np.sqrt((2/3)*(M-1))
    bits = []

    for s in symbols:
        I = s.real * norm
        Q = s.imag * norm
        i_bits = level_to_bits(I, m_side)
        q_bits = level_to_bits(Q, m_side)
        bits.extend(i_bits + q_bits)

    return np.array(bits, dtype=int)

def generate_sequential_bits(M, num_full_constellations=1):
    """Generate bits for full, sequential constellation coverage."""
    k = int(np.log2(M))
    full_bits = []
    for _ in range(num_full_constellations):
        for i in range(M):
            binstr = format(i, f'0{k}b')
            full_bits.extend([int(b) for b in binstr])
    return np.array(full_bits)
def generate_all_points_in_constellation(M):
    k = int(np.log2(M))
    u_seq = generate_sequential_bits(M)  # 64 bits for 16 symbols
    symbols_seq = modulation_MQAM(u_seq, M)
    plot_constellation(symbols_seq, bits_per_symbol=k, title="16-QAM with Gray Mapping", annotate=True)

def plot_constellation(symbols,bits_per_symbol, title="Constellation Diagram", annotate=False):
    """
    Plot the constellation of complex symbols.
    - symbols: 1D array of complex numbers
    - title: plot title
    """
    plt.ion() # interactive mode on so it does not block the following code
    symbols = np.asarray(symbols)  # ensure numpy array
    plt.figure(figsize=(6,6))
    
    # Plot the points
    plt.scatter(symbols.real, symbols.imag, color='blue', s=50)
    
    # Add grid, labels
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.title(title)
    plt.axis('equal')  # make axes scale equal so points look right

    # Annotate points with index (optional)
    if annotate:
        if bits_per_symbol > 6:
            size = 3
        else: 
            size = 8
        for i, s in enumerate(symbols):
            # Convert index i to bit string of length bits_per_symbol
            bit_string = format(i % (2**bits_per_symbol), f'0{bits_per_symbol}b')
            plt.text(round(s.real, 3)-0.05 , round(s.imag,3) + 0.05, bit_string, fontsize = size , color='red')

    plt.show()


