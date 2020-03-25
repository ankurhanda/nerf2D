import matplotlib.pyplot as plt
import numpy as np

W = 256

x = np.linspace(0, W-1, W)
x = (x*1.0 / W)*2 -1 

L = 5

c = ['r', 'g', 'b', 'c', 'm']

for el in range(0, L):
    y = np.sin(2**el*np.pi*x)
    plt.plot(x, y, color=c[el], label=str('L = {}').format(el), linewidth=2)

plt.xlabel('x', fontsize=14)
plt.ylabel('$sin(2^L \pi x$)', fontsize=10)
plt.legend(loc='upper right')
plt.savefig('sin.png', figsize=(10, 5), dpi=100)
plt.show()
