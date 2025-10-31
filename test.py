from M_QAM_modulation import modulation_MQAM, plot_constellation, demodulate_MQAM,generate_all_points_in_constellation
from ASCII import encode_to_binary, decode_from_binary
import numpy as np
from simple_CE import insert_pilots_after_blocks , trim_pilots_from_frame
M = 16
true_H = 0.8 + 0.6j  #trial
pilot_symbol=1+0j
data = "hey i am here heree come ppaababe qwerr kiot mnklzzxa alks jfose aaaaa rejbjld newkfknel nsedfac efanawkE S edsn fd ln as dfflnsdneaaaa"
binary_strings = encode_to_binary(data)
bit_stream = np.array([int(b) for bs in binary_strings for b in bs])



mod_data = modulation_MQAM(bit_stream,M)

#here we need to add the pilot symbols needed to estimate the channel 
data_with_pilots, pilot_p = insert_pilots_after_blocks(mod_data, pilot_symbol, block_size=4)

# True channel (for simulation)
SNR_dB = 20  # Noise level
sigma = 10**(-SNR_dB/20)  # Noise std dev (for unit power signal)
noise = (np.random.randn() + 1j*np.random.randn()) * sigma / np.sqrt(2)

# Simulate n pilots
Rx = np.zeros(len(data_with_pilots), dtype=complex)  # Received vector
y = np.zeros(len(pilot_p), dtype=complex)  # Received vector
index_of_total_data = len(data_with_pilots)
ind_temp = 0
for k in range(len(data_with_pilots)):
    Rx[k] = true_H * data_with_pilots[k] + noise
    if(k == pilot_p[ind_temp]):
        y[ind_temp] = true_H * pilot_symbol + noise
        ind_temp += 1

# LS estimation: Average over pilots
H_est = np.mean(y / pilot_symbol)  # Or explicitly: (1/n) * sum(y_k / s)
print(f"True H: {true_H}")
print(f"Est H (n={len(pilot_p)}): {H_est}")
print(f"Error magnitude: {np.abs(true_H - H_est)}")

# Now we can equalize the received data
equalized_Rx = Rx / H_est
# Now we need to trim the pilots from the equalized received data
trimed =trim_pilots_from_frame(equalized_Rx, pilot_p)

#generate_all_points_in_constellation(M)
#plot_constellation(mod_data, 4,title=str(M)+"-QAM Constellation")

data_ASCII = demodulate_MQAM(trimed, M)
deascii_data = decode_from_binary(data_ASCII)

if data == deascii_data:
    print("Success! Data matches original.")
else:
    print("Failure! Data does not match original.")