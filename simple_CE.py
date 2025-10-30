import numpy as np

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

def trim_pilots_from_frame(rx_frame, pilot_indices):
   
    num_pilots = len(pilot_indices)
    data_len = len(rx_frame) - num_pilots
    data = np.zeros(data_len, dtype=complex)
    
    data_idx = 0
    pilot_set = set(pilot_indices)  # For O(1) lookup
    
    for i in range(len(rx_frame)):
        if i not in pilot_set:
            data[data_idx] = rx_frame[i]
            data_idx += 1
    
    return data
