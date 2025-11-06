import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

# Muda a fonte dos Plots
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'
plt.rcParams['font.family'] = 'lmodern'

# Parâmetros do filtro
K = 4
i = np.arange(1,K+1)

# Calcular os polos do Filtro
Omega_c = 10
poles = 1j*Omega_c*np.exp(1j*np.pi/(2*K)*(2*i-1))

# Determina o polinômio denominador
denominator = np.real(np.poly(poles))
print(denominator)

# Plota a Resposta em Magnitude do Filtro Projetado
w = np.arange(0, 30, 0.01)
H = denominator[-1]/np.polyval(denominator, 1j*w)
plt.figure(figsize=(6,3))
plt.plot(w, 20*np.log10(np.abs(H)), color='black')
plt.xlabel('$\Omega$ (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.xlim(0, 30)
plt.grid()
plt.tight_layout()
plt.show()
