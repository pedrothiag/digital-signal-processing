import numpy as np
import matplotlib.pyplot as plt
import control as ct
np.set_printoptions(suppress=True)

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'
plt.rcParams['font.family'] = 'lmodern'

# Parametros de Projeto
Ts = np.pi/100
Omega_p = 15
alpha_p = 1
Omega_s = 10
alpha_s = 6.3

# Aplicar a compensação de warping para transformação bilinear
Omega_p = np.tan(Omega_p*Ts/2)
Omega_s = np.tan(Omega_s*Ts/2)
# print('-'*80)
# print(f"Omega_p: {Omega_p:.4f}")
# print(f"Omega_s: {Omega_s:.4f}")
# print('-'*80,'\n')

##############################################################
# Determinar o filtro passa-baixas prototipo
##############################################################
# Determinar a frequencia da banda de passagem e rejeição do filtro passa-baixas prototipo
wp = 1
ws = Omega_p/Omega_s

# Determinar a ordem minima do filtro Chebyshev passa-baixas prototipo
K = np.ceil(np.arccosh(np.sqrt((10**(alpha_s/10)-1)/(10**(alpha_p/10)-1))) / np.arccosh(ws/wp)).astype(int)
print('-'*80)
print(f"K = {K:.0f}")
print('-'*80,'\n')

# Calcula os polos do filtro Chebyshev passa-baixas protótipo
epsilon = np.sqrt(10**(alpha_p/10) - 1)
i = np.arange(1,K+1)
lp_poles = -np.sinh(np.arcsinh(1/epsilon)/K) * np.sin(np.pi*(2*i-1)/(2*K)) + 1j*np.cosh(np.arcsinh(1/epsilon)/K) * np.cos(np.pi*(2*i-1)/(2*K))

# Calcula a funcao de transferencia do filtro passa-baixas prototipo
K0 = 1/np.sqrt(1+epsilon**2) if K%2==0 else 1             # Ganho em DC
# lp_den = np.real(np.poly(lp_poles))
# Ka = K0*np.real(np.prod(-lp_poles))                       # Ganho do filtro analogico
# lp_num = Ka                                             # O filtro passa-baixas prototipo nao possui zeros finitos
# print('-'*80)
# print(f'Filtro passa-baixas protótipo:')
# print(f'Numerador: {lp_num}')
# print(f'Denominador: {lp_den}')
# print('-'*80,'\n')

##############################################################
# Determinar o filtro passa-altas analogico
##############################################################
M = 0                                                   # O filtro passa-baixas prototipo nao possui zeros finitos
hp_poles = Omega_p / lp_poles
hp_zeros = np.zeros(int(K - M))
Khp = K0                                                # O ganho do filtro passa-altas e igual ao ganho DC do filtro passa-baixas

##############################################################
# Determinar o filtro passa-altas discretizado
##############################################################
Number_of_Poles = len(hp_poles)
Number_of_Zeros = len(hp_zeros)
Kd = Khp*np.real(np.prod(1 - hp_zeros))/(np.real(np.prod((1 - hp_poles))))           # Ganho do filtro digital
d_poles = (1 + hp_poles) / (1 - hp_poles)
d_zeros_0 = (1 + hp_zeros) / (1 - hp_zeros)
d_zeros_1 = -1.0*np.ones(int(Number_of_Poles - Number_of_Zeros))
d_zeros = np.concatenate((d_zeros_0, d_zeros_1))
d_den = np.real(np.poly(d_poles))
d_num = Kd * np.real(np.poly(d_zeros))
Hd = ct.tf(d_num, d_den, dt=1)
print('-'*80)
print(f'Filtro passa-altas digital:')
# print(f'Ganho Kd: {Kd}')
# print(f'Numerador: {d_num}')
# print(f'Denominador: {d_den}')
print(Hd)
print('-'*80,'\n')


##############################################################
# Plotar a resposta em Frequencia
##############################################################
w = np.arange(0, np.pi, 0.01)
H = np.polyval(d_num, np.exp(1j*w)) / np.polyval(d_den, np.exp(1j*w))

plt.figure(figsize=(6,3))
plt.plot(w/Ts, 20*np.log10(np.abs(H)), color='black')
plt.xlabel('$\Omega$ (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.xlim(0, 30)
plt.grid()
plt.tight_layout()
plt.show()