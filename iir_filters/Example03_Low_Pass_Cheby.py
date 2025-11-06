import numpy as np
import matplotlib.pyplot as plt
import control as ct
np.set_printoptions(suppress=True)

#Muda a fonte dos Plots para Times New Roman
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'
plt.rcParams['font.family'] = 'lmodern'

# Parâmetros do filtro
W_p = 10
a_p = 2
W_s = 16.5
a_s = 20

# Calcula o Epsilon
epsilon = np.sqrt(10**(a_p/10) - 1)

# Calcular a ordem mínima do filtro
K = np.ceil((np.acosh(np.sqrt((10**(a_s/10)-1)/(10**(a_p/10)-1))))/(np.acosh(W_s/W_p))).astype(int)
print(f'Ordem mínima do filtro Chebyshev: K = {K}')

# Calcular os polos do filtro
i = np.arange(1,K+1)
poles = -W_p * np.sinh(np.arcsinh(1/epsilon)/K) * np.sin(np.pi*(2*i-1)/(2*K)) + 1j*W_p * np.cosh(np.arcsinh(1/epsilon)/K) * np.cos(np.pi*(2*i-1)/(2*K))

# Calcular o numerador e denominador da função de transferência
denominator = np.real(np.poly(poles))
Klp = 1/np.sqrt(1+epsilon**2) if K%2==0 else 1
numerator = Klp*np.real(np.prod(-poles))
H = ct.tf(numerator, denominator)
print("Função de Transferência do Filtro Chebyshev:")
print(H)

# Calcula a resposta em magnitude nos pontos Wp e Ws
Hp = numerator/np.polyval(denominator, 1j*W_p)
Hs = numerator/np.polyval(denominator, 1j*W_s)
print(f'Magnitude em Wp = {W_p} rad/s: |H(jWp)| = {20*np.log10(np.abs(Hp)):.2f} dB')
print(f'Magnitude em Ws = {W_s} rad/s: |H(jWs)| = {20*np.log10(np.abs(Hs)):.2f} dB')

# Calcula a resposta em magnitude
w = np.arange(0, 30, 0.01)
H = numerator/np.polyval(denominator, 1j*w)

plt.figure(figsize=(6,3))
plt.plot(w, 20*np.log10(np.abs(H)), color='black')
plt.xlabel('$\Omega$ (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.xlim(0, 30)
plt.grid()
plt.tight_layout()
plt.show()