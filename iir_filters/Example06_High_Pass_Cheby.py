import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'
plt.rcParams['font.family'] = 'lmodern'

Omega_p = 165
Omega_s = 100
alpha_p = 2
alpha_s = 20

# Passo 1 - Filtro Passa-Baixas Protótipo
# 1.1 - Normalização das frequências
Omega_p_prot = 1
Omega_s_prot = Omega_p/Omega_s

# 1.2 - Cálculo da ordem do filtro
K = np.ceil(np.arccosh(np.sqrt((10**(alpha_s/10)-1)/(10**(alpha_p/10)-1)))/np.arccosh(Omega_s_prot/Omega_p_prot)).astype(int)
print(f'Ordem do Filtro (K): {K}')

# 1.3 - Cálculo dos polos do filtro Chebyshev Tipo I
epsilon = np.sqrt(10**(alpha_p/10) - 1)
i = np.arange(1,K+1)
poles_low_pass = -np.sinh(np.arcsinh(1/epsilon)/K) * np.sin(np.pi*(2*i-1)/(2*K)) + 1j*np.cosh(np.arcsinh(1/epsilon)/K) * np.cos(np.pi*(2*i-1)/(2*K))

# 1.4 - Ganho do Filtro Passa-Baixas
K0 = 1/np.sqrt(1+epsilon**2) if K%2==0 else 1

# Passo 2 - Transformação do Filtro Passa-Baixas em Passa-Altas
# 2.1 - Cálculo dos polos do filtro Passa-Altas
poles = Omega_p/poles_low_pass
zeros = np.zeros(K)

# 2.2 - Cálculo do numerador e denominador da função de transferência
den = np.real(np.poly(poles))
num = K0*np.real(np.poly(zeros))
# num = np.zeros(K+1)
# num[0] = Kn
print(f'Numerador: {num}')
print(f'Denominador: {den}')

# Passo 3 - Análise do Filtro Projetado
w = np.arange(0, 300, 0.01)
H = np.polyval(num, 1j*w)/np.polyval(den, 1j*w)

plt.figure(figsize=(6,3))
plt.plot(w, 20*np.log10(np.abs(H)), color='black')
plt.xlabel('$\Omega$ (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.xlim(0, 300)
plt.grid()
plt.tight_layout()
plt.show()