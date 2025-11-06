import numpy as np
import matplotlib.pyplot as plt
import control as ct
np.set_printoptions(suppress=True)

#Muda a fonte dos Plots para Times New Roman
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'
plt.rcParams['font.family'] = 'lmodern'

# Parametros de Projeto
Ts = 0.01
Omega_p = 8*np.pi
alpha_p = 2
Omega_s = 15*np.pi
alpha_s = 11

# Aplicar a compensação de warping para transformação bilinear
Omega_p = np.tan(Omega_p*Ts/2)
Omega_s = np.tan(Omega_s*Ts/2)

# Determina a ordem do filtro Chebyshev
K = np.ceil(np.arccosh(np.sqrt((10**(alpha_s/10)-1)/(10**(alpha_p/10)-1))) / np.arccosh(Omega_s/Omega_p)).astype(int)
print('-'*50)
print(f'Ordem do filtro: K = {K:.0f}')
print('-'*50,'\n')

# Calcula os polos do filtro Chebyshev passa-baixas protótipo
epsilon = np.sqrt(10**(alpha_p/10) - 1)
i = np.arange(1,K+1)
analog_poles = -Omega_p*np.sinh(np.arcsinh(1/epsilon)/K) * np.sin(np.pi*(2*i-1)/(2*K)) + 1j*Omega_p*np.cosh(np.arcsinh(1/epsilon)/K) * np.cos(np.pi*(2*i-1)/(2*K))
# print('-'*50)
# print('Polos do filtro Chebyshev passa-baixas:')
# print(analog_poles)
# print('-'*50,'\n')

# Determina o numerador e denominador do filtro analógico protótipo
K0 = 1/np.sqrt(1+epsilon**2) if K%2==0 else 1
Ka = K0*np.real(np.prod(-analog_poles))
analog_den = np.real(np.poly(analog_poles))
analog_num = Ka
print('-'*50)
print(f'Numerador do filtro analógico: {analog_num}')
print(f'Denominador do filtro analógico: {analog_den}')
print('-'*50,'\n')

#-----------
# Determina os polos, zeros e ganho do filtro digital
# N = len(analog_poles)
M = 0
digital_poles = (1 + analog_poles) / (1 - analog_poles)
digital_zeros = np.array([-1]*(K - M))
Kd = Ka * 1/np.prod((1 - analog_poles))

# Determina o numerador e denominador do filtro digital
digital_den = np.real(np.poly(digital_poles))
digital_num = np.real(Kd*np.poly(digital_zeros))
sysd = ct.tf(digital_num, digital_den, dt=1)
print('-'*50)
print('Função de transferência do filtro digital IIR Chebyshev passa-baixas:')
print(sysd)
print('-'*50,'\n')

# Plota a resposta em frequência do filtro digital projetado
w = np.arange(0, np.pi, 0.01)
H = np.polyval(digital_num, np.exp(1j*w)) / np.polyval(digital_den, np.exp(1j*w))

w0 = (15*np.pi)*Ts
H0 = 20*np.log10(np.abs(np.polyval(digital_num, np.exp(1j*w0)) / np.polyval(digital_den, np.exp(1j*w0))))
print(H0)

plt.figure(figsize=(6,3))
plt.plot(w/Ts, 20*np.log10(np.abs(H)), color='black')
plt.xlabel('$\Omega$ (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.xlim(0, np.pi/Ts)
plt.xticks([0, 20*np.pi, 40*np.pi, 60*np.pi, 80*np.pi, 100*np.pi],
           ['$0$', r'$20\pi$', r'$40\pi$', r'$60\pi$', r'$80\pi$', r'$100\pi$'])
plt.axvline(w0/Ts, color='k', linestyle='--', linewidth=1)
plt.axhline(H0, color='k', linestyle='--', linewidth=1)
plt.annotate(
    f'{H0:.2f} dB',           # texto
    xy=(w0/Ts, H0),                # ponto a ser destacado
    xytext=(w0 + 5, H0 - 30),# posição do texto
    arrowprops=dict(arrowstyle='->', color='k', lw=1),
    color='k',
    fontsize=10
)
plt.grid()
plt.tight_layout()
plt.show()