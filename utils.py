import numpy as np

# -------------------
# Simple LFSR scrambler (self inverse)
# -------------------
def lfsr_sequence(length, seed=1, taps=(31,28)):
    """
    Generate a PN sequence using a simple LFSR (Galois/standard).
    taps: tuple of tap bit positions (1-based) for feedback polynomial.
    seed: integer seed (nonzero) => initial register state (use at most 31 bits).
    Returns array of 0/1 of requested length.
    NOTE: This is NOT the full 3GPP Gold generator but is deterministic and reversible.
    """
    reg = seed & 0x7fffffff
    if reg == 0:
        reg = 1
    seq = np.zeros(length, dtype=np.int8)
    for i in range(length):
        # output bit = LSB of reg
        seq[i] = reg & 1
        # compute feedback XOR of specified taps (convert to 0-based)
        fb = 0
        for t in taps:
            fb ^= (reg >> (t-1)) & 1
        # shift register and feed new bit into MSB
        reg = (reg >> 1) | (fb << 30)  # 31-bit LFSR
    return seq

def scramble_bits(coded_bits, seed=1, taps=(31,28)):
    pn = lfsr_sequence(len(coded_bits), seed=seed, taps=taps)
    return np.bitwise_xor(coded_bits.astype(np.int8), pn.astype(np.int8))

def descramble_bits(rx_bits, seed=1, taps=(31,28)):
    # XOR inverse is same operation
    return scramble_bits(rx_bits, seed=seed, taps=taps)

# -------------------
# Simple rate-matching shorthand (shortening/puncturing stub)
# -------------------
def rate_match_shorten(coded_bits, E):
    """
    Simple shortening/puncturing stub:
    - If E <= len(coded_bits): return first E bits (shortening/truncation)
    - If E > len(coded_bits): pad with zeros (not ideal, but explicit)
    For real NR rate-matching follow the standard interleaver/puncture/repetition rules.
    """
    L = len(coded_bits)
    if E <= L:
        return coded_bits[:E].astype(np.int8)
    else:
        pad = np.zeros(E - L, dtype=np.int8)
        return np.concatenate([coded_bits.astype(np.int8), pad])

def gold_sequence(length, c_init=0, n_start=0, Nc=1600):
    """
    Generate a gold sequence of given length starting at index n_start.
    This follows the common 3GPP/LTE/NR convention:
      x1 feedback: x1(n+31) = x1(n+3) XOR x1(n)
      x2 feedback: x2(n+31) = x2(n+3) XOR x2(n+2) XOR x2(n+1) XOR x2(n)
    Initialization:
      x1[0..30] = 1 (all ones)
      x2[0..30] = bits of c_init (LSB -> x2[0], ... up to bit30)
    Parameters:
      length: number of bits requested
      c_init: integer seed used to init x2 (typical: cell/RNTI-derived)
      n_start: sequence index to start from (default 0)
      Nc: offset (default 1600 per 3GPP convention)
    Returns:
      numpy array of 0/1 bits of size `length`.
    """
    if length <= 0:
        return np.array([], dtype=np.int8)

    # total sequence length we need to produce
    total_needed = n_start + Nc + length

    # initialize x1 (31 bits) to all ones
    x1 = [1] * 31

    # initialize x2 from c_init LSB->x2[0] up to bit30
    x2 = [ ((c_init >> i) & 1) for i in range(31) ]

    seq = np.zeros(total_needed, dtype=np.int8)

    # generate sequence indices 0 .. total_needed-1
    for n in range(total_needed):
        # output is current x1[0] xor x2[0]
        seq[n] = x1[0] ^ x2[0]

        # compute new bits using recurrence (based on current state)
        new_x1 = x1[3] ^ x1[0]  # x1(n+31) = x1(n+3) xor x1(n)
        new_x2 = x2[3] ^ x2[2] ^ x2[1] ^ x2[0]  # x2(n+31) = x2(n+3)^x2(n+2)^x2(n+1)^x2(n)

        # shift: drop oldest (index 0), append new at the end
        x1 = x1[1:] + [new_x1]
        x2 = x2[1:] + [new_x2]

    start_idx = n_start + Nc
    end_idx = start_idx + length
    return seq[start_idx:end_idx].astype(np.int8)


# Example helper to scramble/descramble a bit array in place:
def gold_scramble(bits, c_init=0, n_start=0, Nc=1600):
    """
    XOR the provided bits with the gold sequence starting at n_start.
    bits: numpy array of 0/1 ints (will not be modified in-place; returns new array)
    """
    L = len(bits)
    g = gold_sequence(L, c_init=c_init, n_start=n_start, Nc=Nc)
    return np.bitwise_xor(bits.astype(np.int8), g)

