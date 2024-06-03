import hashlib

import numpy as np


def hash_2d_array(arr: np.ndarray):
    # Convert the array to a bytes object
    arr_bytes = arr.tobytes()

    # Create a hashlib object (you can choose a different hashing algorithm if needed)
    hash_object = hashlib.md5()

    # Update the hash with the array bytes
    hash_object.update(arr_bytes)

    # Get the hexadecimal representation of the hash
    unique_hash = int(hash_object.hexdigest(), 16)

    return unique_hash
