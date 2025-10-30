def encode_to_binary(word):

    return [bin(ord(char))[2:].zfill(8) for char in word]

def decode_from_binary(bits):
    binary_strings_dec = []
    for i in range(0, len(bits), 8):
        chunk = bits[i:i+8]
        binary_str = ''.join(str(bit) for bit in chunk)
        binary_strings_dec.append(binary_str)

    return ''.join(chr(int(b, 2)) for b in binary_strings_dec)

