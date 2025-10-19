import sys
sys.path.append('./polarcodes')  # Add local folder to path if needed
from polarcodes import PolarCode, Construct, Encode, Decode
from OFDM import *
from ASCII import encode_to_binary,decode_from_binary
from utils import *
from modulation import *
from QPSK_AWGN import *
import numpy as np

#data creaition 
data = "hey i am here"
binary_strings = encode_to_binary(data)
bit_stream = np.array([int(b) for bs in binary_strings for b in bs])
print(f"[DEBUG] Original bit_stream length: {len(bit_stream)}")

# CRC is next 16CRC is used for less than 3000 for more than 3000 24CRC is used
bit_stream_crc = attach_crc(bit_stream)
print(f"[DEBUG] bit_stream_crc length: {len(bit_stream_crc)} (added CRC)")

# polar code paramiters and encoding using a library 
K = len(bit_stream_crc)  
N = 1024  # we are working on a control system and the max lenght is 1024 
design_SNR = 5.0  # FIXED: Higher for recovery

myPC = PolarCode(N, K)  
myPC.construction_type = 'bb'  
Construct(myPC, design_SNR)

myPC.set_message(bit_stream_crc.astype(np.int32))
Encode(myPC)
coded_bits = myPC.get_codeword()
print(f"[DEBUG] coded_bits length: {len(coded_bits)}, first 10: {coded_bits[:10]}")

# Scrambling Stub (DISABLED for test)
scramble_init = 0
coded_bits_scrambled = coded_bits.copy()  # FIXED: No scramble
print(f"[DEBUG] After scramble, first 10: {coded_bits_scrambled[:10]}")

# Rate Matching Stub (FIXED: E=N for test, no shortening)
E = N  # FIXED: Full for debug
# E = 900  # Uncomment for shortening
coded_bits_matched = rate_match_shorten(coded_bits_scrambled, E)
print(f"[DEBUG] After rate matching, length: {len(coded_bits_matched)}, first 10: {coded_bits_matched[:10]}")

# Modulation
mod_order = 'QPSK'
modulated_symbols = nr_modulate(coded_bits_matched, mod_order)
print(f"[DEBUG] modulated_symbols length: {len(modulated_symbols)}, first 2: {modulated_symbols[:2]}")

# OFDM TX
tx_ofdm = ofdm_tx(modulated_symbols)
print(f"[DEBUG] tx_ofdm length: {len(tx_ofdm)}, first 10 real: {np.real(tx_ofdm[:10])}")

# Multipath Channel (FIXED: Flat for test)
cir = [1.0]  # FIXED: Flat
delays = [0]
# cir = [1.0, 0.5 * np.exp(-1j * np.pi / 4), 0.3 * np.exp(1j * np.pi / 6)]  # Uncomment for multipath
# delays = [0, 16, 32]
rx_ofdm = multipath_channel(tx_ofdm, cir, delays)
print(f"[DEBUG] rx_ofdm length: {len(rx_ofdm)}, first 10 real: {np.real(rx_ofdm[:10])}")

# OFDM RX
cir_time = np.zeros(N_fft, dtype=complex)  # FIXED: Full N_fft, complex
for d, tap in zip(delays, cir):
    if d < N_fft:
        cir_time[d] = tap
cir_freq = fft(cir_time)  # FIXED: Full FFT
print(f"[DEBUG] cir_freq shape: {cir_freq.shape}, first 10: {cir_freq[:10]}")
rx_symbols = ofdm_rx(rx_ofdm, cir_freq)
print(f"[DEBUG] rx_symbols length: {len(rx_symbols)}, first 2: {rx_symbols[:2]}")

rx_symbols = rx_symbols[:len(modulated_symbols)]  # Trim

# PN for descrambling (DISABLED for test)
pn = np.zeros(N, dtype=int)  # FIXED: All 0
print(f"[DEBUG] pn length: {len(pn)}, first 10: {pn[:10]}")

# QPSK_AWGN for LLRs
qpsk_channel = QPSK_AWGN(myPC, rx_symbols, design_SNR)
llrs = myPC.likelihoods.copy()
print(f"[DEBUG] llrs length before pad: {len(llrs)}, min/max: {np.min(llrs):.2f}/{np.max(llrs):.2f}")

# FIXED: Pad to N with 10.0 (less bias)
pad_len = N - len(llrs)
llrs = np.pad(llrs, (0, pad_len), mode='constant', constant_values=10.0)  # FIXED: 10.0

# Soft descrambling
llrs = np.where(pn == 1, -llrs, llrs)
print(f"[DEBUG] llrs length after descramble: {len(llrs)}, min/max: {np.min(llrs):.2f}/{np.max(llrs):.2f}")

# FIXED: Clip LLRs to avoid NaN in Decode
llrs = np.clip(llrs, -10, 10)

myPC.likelihoods = llrs

# Decode (unchanged)
Decode(myPC)
decoded_full = myPC.message_received
print(f"[DEBUG] decoded_full length: {len(decoded_full)}, first 10: {decoded_full[:10]} (expected from crc: {bit_stream_crc[:10]})")
print(f"[DEBUG] Bit match rate: {np.mean(decoded_full == bit_stream_crc)}")  # FIXED: % match

# CRC check & string decode (unchanged)
crc_passed, decoded_info_bits, crc_len = check_crc_and_strip(decoded_full)
print(f"[DEBUG] CRC passed: {crc_passed}, inferred CRC len: {crc_len}, info bits len: {len(decoded_info_bits) if decoded_info_bits is not None else 'None'}")

if crc_passed:
    binary_strings_dec = []
    for i in range(0, len(decoded_info_bits), 8):
        chunk = decoded_info_bits[i:i+8]
        binary_str = ''.join(str(bit) for bit in chunk)
        binary_strings_dec.append(binary_str)
    
    decoded_data = decode_from_binary(binary_strings_dec)
    print(f"Decoded string: '{decoded_data}'")
else:
    print("CRC failed—multipath/scrambling errors; try flat CIR or no scramble.")