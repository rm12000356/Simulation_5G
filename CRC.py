import numpy as np

def _poly_to_bits(poly_int, degree):
    """
    Convert generator polynomial (integer form, e.g. 0x1021 for CRC-16-CCITT)
    to a bit-array of length (degree+1) MSB-first, including the leading 1.
    """
    bits = np.zeros(degree + 1, dtype=np.int8)
    for i in range(degree + 1):
        # bit index from MSB (degree) down to 0
        bits[i] = (poly_int >> (degree - i)) & 1
    return bits

def _compute_crc_bits(message_bits, poly_int, degree):
    """
    Compute CRC bits (length = degree) using polynomial division (MSB-first).
    message_bits: 1D numpy array of 0/1 ints
    poly_int: integer representation of generator poly (e.g. 0x1021)
    degree: CRC degree L (16 or 24)
    Returns crc_bits (length degree)
    """
    msg = np.concatenate([message_bits.astype(np.int8), np.zeros(degree, dtype=np.int8)])
    gen = _poly_to_bits(poly_int, degree)  # length degree+1
    A = len(message_bits)
    for i in range(A):
        if msg[i] == 1:
            # XOR generator across msg[i:i+degree+1]
            msg[i:i + degree + 1] ^= gen
    crc = msg[A:A+degree].copy()
    return crc

def attach_crc(message_bits):
    """
    Attach CRC16 (<=3824 bits) or CRC24B (>3824).
    message_bits: numpy array of 0/1 ints
    returns concatenated array of length A+L
    """
    A = len(message_bits)
    if A > 3824:
        crc_len = 24
        poly = 0x864CFB  # CRC-24/B (generator integer)
    else:
        crc_len = 16
        poly = 0x1021    # CRC-16-CCITT
    crc = _compute_crc_bits(message_bits, poly, crc_len)
    return np.concatenate([message_bits.astype(np.int8), crc.astype(np.int8)])

def _check_crc_bits(full_bits, poly_int, degree):
    """
    Check that CRC bits are correct: run division over full_bits and
    confirm remainder == 0
    full_bits length = A + degree
    """
    msg = full_bits.copy().astype(np.int8)
    total = len(msg)
    A = total - degree
    gen = _poly_to_bits(poly_int, degree)
    for i in range(A):
        if msg[i] == 1:
            msg[i:i + degree + 1] ^= gen
    remainder = msg[A:A+degree]
    return np.all(remainder == 0)

def check_crc_and_strip(decoded_full):
    """
    Try to infer CRC type by length and check CRC. Returns (crc_passed, info_bits, crc_len)
    If none match returns (False, None, 0)
    """
    total = len(decoded_full)
    # Try both possibilities but infer by length threshold
    # If total <= 3824+16 => likely CRC16; if total > 3824+16 => CRC24
    # We'll attempt based on thresholds but still test rigorously.
    if total <= 3824 + 16:
        # try CRC16
        if total < 16:
            return False, None, 0
        if _check_crc_bits(decoded_full, 0x1021, 16):
            return True, decoded_full[:total - 16].astype(np.int8), 16
        else:
            return False, None, 16
    else:
        # prefer CRC24 if length allows
        if total < 24:
            return False, None, 0
        if _check_crc_bits(decoded_full, 0x864CFB, 24):
            return True, decoded_full[:total - 24].astype(np.int8), 24
        # fallback: maybe CRC16
        if _check_crc_bits(decoded_full, 0x1021, 16):
            return True, decoded_full[:total - 16].astype(np.int8), 16
        return False, None, 0
