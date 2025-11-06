import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'
plt.rcParams['font.family'] = 'lmodern'

Omega_p = 10
Omega_s = 20
alpha_p = 2
alpha_s = 20

# Determinar os polos do filtro Chebyshev Tipo I
K = np.ceil(np.arccosh(np.sqrt((10**(alpha_s/10)-1)/(10**(alpha_p/10)-1)))/np.arccosh(Omega_s/Omega_p)).astype(int)
print(f'Ordem do Filtro (K): {K}')

epsilon = 1/np.sqrt(10**(alpha_s/10) - 1)       #Epsilon calculado a partir da atenuação na banda de rejeição
i = np.arange(1,K+1)
poles = -Omega_p * np.sinh(np.arcsinh(1/epsilon)/K) * np.sin(np.pi*(2*i-1)/(2*K)) + 1j*Omega_p * np.cosh(np.arcsinh(1/epsilon)/K) * np.cos(np.pi*(2*i-1)/(2*K))

# Determinar os polos e zeros do Filtro de Chebyshev Inverso
poles = Omega_p*Omega_s/poles
zeros = 1j*Omega_s*1/np.cos(np.pi*(2*i-1)/(2*K))

# Determinar o numerador e denominador da função de transferência
numerator = np.real(np.prod(poles/zeros)*np.poly(zeros))
denominator = np.real(np.poly(poles))
print(f'Numerador: {numerator}')
print(f'Denominador: {denominator}')

w = np.arange(0, 80, 0.01)
H = np.polyval(numerator, 1j*w)/np.polyval(denominator, 1j*w)

plt.figure(figsize=(6,3))
plt.plot(w, np.abs(H), color='black')
plt.xlabel('$\Omega$ (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.xlim(0, np.max(w))
plt.grid()
plt.tight_layout()
plt.show()