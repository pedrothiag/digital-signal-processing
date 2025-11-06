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
alpha_p = 2.4
alpha_s = 20

# Passo 1 - Determinar os polos do filtro passa-baixas prototipo
# Calculo das Frequências do Passa-Baixas
Omega_p = 1
Omega_s_1 = (Omega_p1*Omega_p2 - Omega_s1**2)/(Omega_s1*(Omega_p2 - Omega_p1))
Omega_s_2 = (Omega_s2**2 - Omega_p1*Omega_p2)/(Omega_s2*(Omega_p2 - Omega_p1))
Omega_s = min(abs(Omega_s_1), abs(Omega_s_2))
# print('-'*80)
# print(f'Omega_s: {Omega_s:.4f}')
# print('-'*80,'\n')

# Determina a ordem do filtro de Butterworth
K = np.ceil(np.log10((10**(alpha_s/10)-1)/(10**(alpha_p/10)-1)) / (2*np.log10(Omega_s/Omega_p))).astype(int)
print('-'*80)
print(f'Ordem do filtro: K = {K:.0f}')
print('-'*80,'\n')

# Determina a Frequencia de Corte do Filtro Butterworth (atendendo a rejeição)
Omega_c = Omega_s / ((10**(alpha_s/10) - 1)**(1/(2*K)))
# print('-'*80)
# print(f'Freq. de Corte: wc = {Omega_c:.4f}')
# print('-'*80,'\n')

# Calcula os polos do filtro Butterworth passa-baixas protótipo
i = np.arange(1,K+1)
poles = 1j*Omega_c*np.exp(1j*np.pi/(2*K)*(2*i-1))

# Determina o numerador e denominador do filtro analógico protótipo
K0 = 1
# num = Kn*np.real(np.prod(-poles))
# den = np.real(np.poly(poles))
# print('-'*80)
# print(f'Numerador do filtro passa-baixas: {num}')
# print(f'Denominador do filtro passa-baixas: {den}')
# print('-'*80,'\n')

# Passo 2 - Transformação dos polos
# Determina os polos e zeros do Filtro Passa-Faixas
P1 = poles*(Omega_p2 - Omega_p1)/2 + np.sqrt((poles*(Omega_p2 - Omega_p1)/2)**2 - Omega_p1*Omega_p2)
P2 = poles*(Omega_p2 - Omega_p1)/2 - np.sqrt((poles*(Omega_p2 - Omega_p1)/2)**2 - Omega_p1*Omega_p2)
Zeros = np.zeros(int(K))

# Determina o numerador e denominador do filtro analógico passa-faixas
Ka = (K0*np.real(np.prod(-poles)))*(Omega_p2 - Omega_p1)**K;
den_PF = np.real(np.poly(np.concatenate((P1, P2))))
num_PF = Ka * np.real(np.poly(Zeros))
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