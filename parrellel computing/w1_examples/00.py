import numpy as np

a = np.ones(3)
b = [1, 1, 1]
print(a)
print(b)

def func(val):
    val[0] = 2

func(a[:])
func(b)
print(a)
print(b)

