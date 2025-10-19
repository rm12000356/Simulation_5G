def encode_to_binary(word):

    return [bin(ord(char))[2:].zfill(8) for char in word]

def decode_from_binary(binary_list):

    return ''.join(chr(int(b, 2)) for b in binary_list)
