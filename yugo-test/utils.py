def array_f4(array):
    for i in range(len(array)):
        array[i] = float(f"{array[i]:.4f}")
    return array
