import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

#Muda a fonte dos Plots para Times New Roman
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'
plt.rcParams['font.family'] = 'lmodern'

# Parametros de Projeto do Filtro Passa-Faixas
Omega_p1 = 1000
Omega_p2 = 2000
Omega_s1 = 450
Omega_s2 = 4000
alpha_p = 1
alpha_s = 20

# Passo 1 - Determinar os polos e zeros do filtro passa-baixas prototipo
#1.1 - Transformação de Frequência Passa-Faixas para Passa-Baixas
Omega_p = 1
Omega_s_1 = (Omega_p1*Omega_p2 - Omega_s1**2)/(Omega_s1*(Omega_p2 - Omega_p1))
Omega_s_2 = (Omega_s2**2 - Omega_p1*Omega_p2)/(Omega_s2*(Omega_p2 - Omega_p1))
Omega_s = min(abs(Omega_s_1), abs(Omega_s_2))
# print('-'*80)
# print(f'Omega_s: {Omega_s:.4f}')
# print('-'*80,'\n')

#1.2 - Determina a ordem do filtro de Chebyshev
K = np.ceil(np.arccosh(np.sqrt((10**(alpha_s/10)-1)/(10**(alpha_p/10)-1))) / np.arccosh(Omega_s/Omega_p))
print('-'*80)
print(f'Ordem do filtro: K = {K:.0f}')
print('-'*80,'\n')

#1.3 - Calcula os polos do filtro Chebyshev passa-baixas protótipo
epsilon = np.sqrt(10**(alpha_p/10) - 1)
# print('-'*80)
# print(f'Epsilon: {epsilon:.4f}')
# print('-'*80,'\n')
i = np.arange(1,K+1)
poles = -np.sinh(np.arcsinh(1/epsilon)/K) * np.sin(np.pi*(2*i-1)/(2*K)) + 1j*np.cosh(np.arcsinh(1/epsilon)/K) * np.cos(np.pi*(2*i-1)/(2*K))

#1.4 - Determina o ganho
K0 = 1/np.sqrt(1+epsilon**2) if K%2==0 else 1
# num = Kn*np.real(np.prod(-poles))
# den = np.real(np.poly(poles))
# print('-'*80)
# print(f'Numerador do filtro passa-baixas: {num}')
# print(f'Denominador do filtro passa-baixas: {den}')
# print('-'*80,'\n')

# Passo 2 - Transformação dos polos
#2.1 - Determina os polos e zeros do Filtro Passa-Faixas
P1 = poles*(Omega_p2 - Omega_p1)/2 + np.sqrt((poles*(Omega_p2 - Omega_p1)/2)**2 - Omega_p1*Omega_p2)
P2 = poles*(Omega_p2 - Omega_p1)/2 - np.sqrt((poles*(Omega_p2 - Omega_p1)/2)**2 - Omega_p1*Omega_p2)
Zeros = np.zeros(int(K))

#2.2 - Determina o numerador e denominador do filtro analógico passa-faixas
den_PF = np.real(np.poly(np.concatenate((P1, P2))))
num_PF = (K0*np.real(np.prod(-poles)))*(Omega_p2 - Omega_p1)**K * np.real(np.poly(Zeros))
print('-'*80)
print(f'Numerador do filtro passa-faixas: {num_PF}')
print(f'Denominador do filtro passa-faixas: {den_PF}')
print('-'*80,'\n')

# Plota a resposta em frequência do filtro passa-faixas
w = np.arange(200, 5000, 0.01)
H = np.polyval(num_PF, 1j*w)/np.polyval(den_PF, 1j*w)

plt.figure(figsize=(6,3))
plt.plot(w, 20*np.log10(np.abs(H)), color='black')
plt.xlabel('$\Omega$ (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.xlim(np.min(w), np.max(w))
plt.grid()
plt.tight_layout()
plt.show()