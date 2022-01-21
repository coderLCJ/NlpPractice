import torch

print('''  _   _      _ _    __        __         _     _ 
 | | | | ___| | | __\\ \\      / /__  _ __| | __| |
 | |_| |/ _ \\ | |/ _ \\ \\ /\\ / / _ \\| '__| |/ _` |
 |  _  |  __/ | | (_) \\ V  V / (_) | |  | | (_| |
 |_| |_|\\___|_|_|\\___/ \\_/\\_/ \\___/|_|  |_|\\__,_|
''')
from math import *
import torch
import numpy as np

def sign(x):
    return np.array([1 if i > 0 else -1 for i in x])

def G1(x):
    return np.array([1 if i < 2.5 else -1 for i in x])

def G2(x):
    return np.array([1 if i < 8.5 else -1 for i in x])

def G3(x):
    return np.array([1 if i < 5.5 else -1 for i in x])

def G(x):
    return 0.4326 * G1(x) * 0.6496 * G2(x) * 0.7514 * G3(x)


x = np.arange(10)
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
w = np.array([0.07143, 0.07143, 0.07143, 0.07143, 0.07143, 0.07143, 0.16667, 0.16667, 0.16667, 0.16667])


print(sign(G(x)))