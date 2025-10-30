import numpy as np

# Parameters
n_pilots = 16
pilot = 1 + 0j  # Simple known pilot symbol (no scrambling needed)
true_H = 0.8 + 0.6j  # True channel (for simulation)
SNR_dB = 20  # Noise level
sigma = 10**(-SNR_dB/20)  # Noise std dev (for unit power signal)

# Simulate n pilots
y = np.zeros(n_pilots, dtype=complex)  # Received vector
for k in range(n_pilots):
    noise = (np.random.randn() + 1j*np.random.randn()) * sigma / np.sqrt(2)
    y[k] = true_H * pilot + noise

# LS estimation: Average over pilots
H_est = np.mean(y / pilot)  # Or explicitly: (1/n) * sum(y_k / s)
print(f"True H: {true_H}")
print(f"Est H (n={n_pilots}): {H_est}")
print(f"Error magnitude: {np.abs(true_H - H_est)}")


#combine pilots witht the data 
def insert_pilots_after_blocks(data, pilot_symbol, block_size):
    data_len = len(data)
    num_full_blocks = data_len // block_size
    num_pilots = num_full_blocks + 1 if data_len % block_size != 0 else num_full_blocks  # one [pilot after each block aka 8 bits aka ASCII code]
    frame_len = data_len + num_pilots
    
    pilot_indices = np.zeros(num_pilots, dtype=int)
    tx_frame = np.zeros(frame_len, dtype=complex)
    data_idx = 0
    frame_idx = 0
    pilot_idx = 0
    
    for block in range(num_full_blocks + 1):
        # Add block_size data (or remainder)
        for _ in range(block_size):
            if data_idx < data_len:
                tx_frame[frame_idx] = data[data_idx]
                data_idx += 1
                frame_idx += 1
            else:
                break
        
        # Add pilot after block (skip if no more data)
        if block < num_pilots:
            pilot_indices[pilot_idx] = frame_idx
            tx_frame[frame_idx] = pilot_symbol
            frame_idx += 1
            pilot_idx += 1
            
    
    return tx_frame , pilot_indices
