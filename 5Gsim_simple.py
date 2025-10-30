import sys
sys.path.append('./polarcodes')  # Add local folder
from polarcodes import PolarCode, Construct, Encode, Decode,AWGN
from ASCII import encode_to_binary, decode_from_binary
from utils import attach_crc, check_crc_and_strip  # Your CRC
import numpy as np

# Data creation
data = "hey i am here heree"
binary_strings = encode_to_binary(data)
bit_stream = np.array([int(b) for bs in binary_strings for b in bs])


# CRC (16 for <3000 bits)
bit_stream_crc = attach_crc(bit_stream)


# Polar params
K = len(bit_stream_crc)
N = 1024 # need bigger N when K is greater than 512
design_SNR = 2.0


myPC = PolarCode(N, K)
myPC.construction_type = 'bb'
Construct(myPC, design_SNR)

myPC.set_message(bit_stream_crc.astype(np.int64))
Encode(myPC)


# AWGN with QPSK (no noise for test; uncomment in AWGN.py for noisy)
AWGN(myPC, design_SNR, mode='QPSK')

# Decode (SCD)
Decode(myPC)
decoded_full = myPC.message_received

print(f"[DEBUG] Bit match rate: {np.mean(decoded_full == bit_stream_crc):.2%}")

# CRC check & string decode
crc_passed, decoded_info_bits, crc_len = check_crc_and_strip(decoded_full)


if crc_passed:    
    decoded_data = decode_from_binary(decoded_info_bits)
    print(f"Decoded string: '{decoded_data}'")
    print(f"Matches original: {decoded_data == data}")
else:
    print("CRC failed—try higher SNR (10.0) or flat channel.")
    

