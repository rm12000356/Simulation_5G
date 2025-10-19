import sys
sys.path.append('./polarcodes')  # Add local folder
from polarcodes import PolarCode, Construct, Encode, Decode
from ASCII import encode_to_binary, decode_from_binary
from utils import attach_crc, check_crc_and_strip  # Your CRC
from modulation import nr_modulate  # Your corrected mod (from history)
from QPSK_AWGN import QPSK_AWGN  # Your adapted AWGN
import numpy as np

# Data creation
data = "hey i am here more text "
binary_strings = encode_to_binary(data)
bit_stream = np.array([int(b) for bs in binary_strings for b in bs])
print(f"[DEBUG] Original bit_stream length: {len(bit_stream)}")

# CRC (16 for <3000 bits)
bit_stream_crc = attach_crc(bit_stream)
print(f"[DEBUG] bit_stream_crc length: {len(bit_stream_crc)}")

# Polar params
K = len(bit_stream_crc)
N = 1024 # need bigger N when K is greater than 512
design_SNR = 0.0  

myPC = PolarCode(N, K)
myPC.construction_type = 'bb'
Construct(myPC, design_SNR)

myPC.set_message(bit_stream_crc.astype(np.int32))
Encode(myPC)
coded_bits = myPC.get_codeword()
print(f"[DEBUG] coded_bits length: {len(coded_bits)}, first 10: {coded_bits[:10]}")

# Modulation (QPSK)
mod_order = 'QPSK'
modulated_symbols = nr_modulate(coded_bits, mod_order)
print(f"[DEBUG] modulated_symbols length: {len(modulated_symbols)}, first 2: {modulated_symbols[:2]}")

# Channel: Simple flat AWGN on symbols
snr_db = 2.0

# QPSK_AWGN for LLRs (adapted for soft)
qpsk_channel = QPSK_AWGN(myPC, modulated_symbols, snr_db)
llrs = myPC.likelihoods.copy()
print(f"[DEBUG] llrs length: {len(llrs)}, min/max: {np.min(llrs):.2f}/{np.max(llrs):.2f}")

# FIXED: Clip LLRs to avoid overflow in Decode (library logdomain)
llrs = np.clip(llrs, -10, 10)  # Finite, stable
myPC.likelihoods = llrs

# Decode (SCD)
Decode(myPC)
decoded_full = myPC.message_received
print(f"[DEBUG] decoded_full length: {len(decoded_full)}, first 10: {decoded_full[:10]} (expected: {bit_stream_crc[:10]})")
print(f"[DEBUG] Bit match rate: {np.mean(decoded_full == bit_stream_crc):.2%}")

# CRC check & string decode
crc_passed, decoded_info_bits, crc_len = check_crc_and_strip(decoded_full)
print(f"[DEBUG] CRC passed: {crc_passed}, CRC len: {crc_len}")

if crc_passed:
    binary_strings_dec = []
    for i in range(0, len(decoded_info_bits), 8):
        chunk = decoded_info_bits[i:i+8]
        binary_str = ''.join(str(bit) for bit in chunk)
        binary_strings_dec.append(binary_str)
    
    decoded_data = decode_from_binary(binary_strings_dec)
    print(f"Decoded string: '{decoded_data}'")
    print(f"Matches original: {decoded_data == data}")
else:
    print("CRC failed—try higher SNR (10.0) or flat channel.")
    