import numpy as np
import matplotlib.pyplot as plt
import control as ct
np.set_printoptions(suppress=True)

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'
plt.rcParams['font.family'] = 'lmodern'

# Parametros de Projeto
Ts = np.pi/10000
Omega_p1 = 1000
Omega_p2 = 2000
Omega_s1 = 450
Omega_s2 = 4000
alpha_p = 2.1
alpha_s = 20

# Aplicar a compensação de warping para transformação bilinear
Omega_p1 = np.tan(Omega_p1*Ts/2)
Omega_p2 = np.tan(Omega_p2*Ts/2)
Omega_s1 = np.tan(Omega_s1*Ts/2)
Omega_s2 = np.tan(Omega_s2*Ts/2)
# print('-'*80)
# print(f"Omega_p1: {Omega_p1:.4f}")
# print(f"Omega_p2: {Omega_p2:.4f}")
# print(f"Omega_s1: {Omega_s1:.4f}")
# print(f"Omega_s2: {Omega_s2:.4f}")
# print('-'*80,'\n')

# ##############################################################
# # Determinar o filtro passa-baixas prototipo
# ##############################################################
# Determinar Omega_s
w_p = 1
w_s_1 = (Omega_p1*Omega_p2 - Omega_s1**2)/(Omega_s1*(Omega_p2 - Omega_p1))
w_s_2 = (Omega_s2**2 - Omega_p1*Omega_p2)/(Omega_s2*(Omega_p2 - Omega_p1))
w_s = min(abs(w_s_1), abs(w_s_2))
# print('-'*80)
# print(f'w_s: {w_s:.4f}')
# print('-'*80,'\n')

# Determina a ordem do filtro de Butterworth
K = np.ceil(np.log10((10**(alpha_s/10)-1)/(10**(alpha_p/10)-1)) / (2*np.log10(w_s/w_p))).astype(int)
print('-'*80)
print(f'Ordem do filtro: K = {K:.0f}')
print('-'*80,'\n')

# Determina a Frequencia de Corte do Filtro Butterworth - Atendendo a rejeicao
Omega_c = w_s / ((10**(alpha_s/10) - 1)**(1/(2*K)))
# print('-'*80)
# print(f'Freq. de Corte: wc = {Omega_c:.4f}')
# print('-'*80,'\n')

# Calcula os polos do filtro Butterworth passa-baixas protótipo
i = np.arange(1,K+1)
poles = 1j*Omega_c * np.exp(1j*np.pi/(2*K)*(2*i-1))

# Determina o numerador e denominador do filtro analógico protótipo
K0 = 1
# Ka = K0 * np.real(np.prod(-poles))
# num = Ka
# den = np.real(np.poly(poles))
# print('-'*80)
# print(f'Numerador do filtro passa-baixas: {num}')
# print(f'Denominador do filtro passa-baixas: {den}')
# print('-'*80,'\n')

##############################################################
# Determinar o filtro passa-faixas analogico
##############################################################
bp_poles1 = poles*(Omega_p2 - Omega_p1)/2 + np.sqrt((poles*(Omega_p2 - Omega_p1)/2)**2 - Omega_p1*Omega_p2)
bp_poles2 = poles*(Omega_p2 - Omega_p1)/2 - np.sqrt((poles*(Omega_p2 - Omega_p1)/2)**2 - Omega_p1*Omega_p2)
bp_poles = np.concatenate((bp_poles1, bp_poles2))
bp_zeros = np.zeros(int(K))
Ka = K0 * np.real(np.prod(-poles)) * (Omega_p2 - Omega_p1)**K

##############################################################
# Determinar o filtro passa-altas discretizado
##############################################################
Number_of_Poles = len(bp_poles)
Number_of_Zeros = len(bp_zeros)
Kd = Ka*np.real(np.prod(1 - bp_zeros))/(np.real(np.prod((1 - bp_poles))))           # Ganho do filtro digital
d_poles = (1 + bp_poles) / (1 - bp_poles)
d_zeros_0 = (1 + bp_zeros) / (1 - bp_zeros)
d_zeros_1 = -1.0*np.ones(int(Number_of_Poles - Number_of_Zeros))
d_zeros = np.concatenate((d_zeros_0, d_zeros_1))
d_den = np.real(np.poly(d_poles))
d_num = Kd * np.real(np.poly(d_zeros))
Hd = ct.tf(d_num, d_den, dt=1)
print('-'*80)
print(f'Filtro passa-faixas digital:')
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
plt.xlim(200, 5000)
plt.ylim(-40,2)
plt.grid()
plt.tight_layout()
plt.show()