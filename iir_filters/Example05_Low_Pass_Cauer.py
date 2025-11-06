import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ellipord, ellip, freqs
np.set_printoptions(suppress=True)

# Especificações do filtro
wp = 10        # frequência de passagem (rad/s)
ws = 25        # frequência de rejeição (rad/s)
rp = 2         # ripple na banda de passagem (dB)
rs = 35        # atenuação mínima na banda de rejeição (dB)

# Cálculo da ordem mínima e frequência de corte ajustada
N, _ = ellipord(wp, ws, rp, rs, analog=True)
print(f"Ordem mínima do filtro: {N}")

# Projeto do filtro elíptico
b, a = ellip(N, rp, rs, wp, btype='low', analog=True)
print(f"Coeficientes do numerador (b): {b}")
print(f"Coeficientes do denominador (a): {a}")

# Resposta em frequência
w = np.arange(0, 40, 0.01)
H = np.polyval(b, 1j*w) / np.polyval(a, 1j*w)

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'
plt.rcParams['font.family'] = 'lmodern'

plt.figure(figsize=(6,3))
plt.plot(w, 20*np.log10(np.abs(H)), color='black')
plt.xlabel('$\Omega$ (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.xlim(0, 40)
plt.grid()
plt.tight_layout()
plt.show()