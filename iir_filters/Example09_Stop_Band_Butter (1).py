import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

#Muda a fonte dos Plots para Times New Roman
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'
plt.rcParams['font.family'] = 'lmodern'

# Parametros de Projeto do Filtro Passa-Faixas
Omega_p1 = 60
Omega_p2 = 260
Omega_s1 = 100
Omega_s2 = 150
alpha_p = 2.2
alpha_s = 20

# Determina o Filtro Passa-Baixas Prototipo
Omega_p = 1
Omega_s_1 = (Omega_s1*(Omega_p2 - Omega_p1))/(Omega_p1*Omega_p2 - Omega_s1**2)
Omega_s_2 = (Omega_s2*(Omega_p2 - Omega_p1))/(Omega_s2**2 - Omega_p1*Omega_p2)
Omega_s = min(abs(Omega_s_1), abs(Omega_s_2))
print('-'*80)
print(f'Omega_s: {Omega_s:.4f}')
print('-'*80,'\n')

# Determina a ordem do filtro de Butterworth
K = np.ceil(np.log10((10**(alpha_s/10)-1)/(10**(alpha_p/10)-1)) / (2*np.log10(Omega_s/Omega_p)))
print('-'*80)
print(f'Ordem do filtro: K = {K:.0f}')
print('-'*80,'\n')

# Determina a Frequencia de Corte do Filtro Butterworth - Atentendo passagem
Omega_c = Omega_p / ((10**(alpha_p/10) - 1)**(1/(2*K)))
print('-'*80)
print(f'Freq. de Corte: wc = {Omega_c:.4f}')
print('-'*80,'\n')

# Calcula os polos do filtro Butterworth passa-baixas protótipo
i = np.arange(1,K+1)
poles = 1j*Omega_c * np.exp(1j*np.pi/(2*K)*(2*i-1))

# Determina o numerador e denominador do filtro analógico protótipo
Kn = 1
num = Kn*np.real(np.prod(-poles))
den = np.real(np.poly(poles))
print('-'*80)
print(f'Numerador do filtro passa-baixas: {num}')
print(f'Denominador do filtro passa-baixas: {den}')
print('-'*80,'\n')

# Determina os polos e zeros do Filtro Rejeita-Faixas
P1 = (Omega_p2 - Omega_p1)/(2*poles) + np.sqrt(((Omega_p2 - Omega_p1)/(2*poles))**2 - Omega_p1*Omega_p2)
P2 = (Omega_p2 - Omega_p1)/(2*poles) - np.sqrt(((Omega_p2 - Omega_p1)/(2*poles))**2 - Omega_p1*Omega_p2)
Z1 = 1j*np.sqrt(Omega_p1*Omega_p2)
Z2 = -1j*np.sqrt(Omega_p1*Omega_p2)

# Determina o numerador e denominador do filtro analógico rejeita-faixas
den_RF = np.real(np.poly(np.concatenate((P1, P2))))
num_RF = Kn*np.real(np.poly(np.concatenate((Z1*np.ones(int(K)), Z2*np.ones(int(K))))))
print('-'*80)
print(f'Numerador do filtro rejeita-faixas: {num_RF}')
print(f'Denominador do filtro rejeita-faixas: {den_RF}')
print('-'*80,'\n')

# Plota a resposta em frequência do filtro rejeita-faixas
w = np.arange(20, 300, 0.01)
H = np.polyval(num_RF, 1j*w)/np.polyval(den_RF, 1j*w)

plt.figure(figsize=(6,3))
plt.plot(w, 20*np.log10(np.abs(H)), color='black')
plt.xlabel('$\Omega$ (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.xlim(np.min(w), np.max(w))
plt.grid()
plt.tight_layout()
plt.show()