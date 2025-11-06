import numpy as np
import matplotlib.pyplot as plt
import control as ct
np.set_printoptions(suppress=True)

#Muda a fonte dos Plots
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'
plt.rcParams['font.family'] = 'lmodern'

# Parâmetros do filtro
W_p = 10
a_p = 2
W_s = 30
a_s = 20

# Determinar a ordem mínima do filtro
K = np.ceil((np.log10((10**(a_s/10)-1)/(10**(a_p/10)-1)))/(2*np.log10(W_s/W_p))).astype(int)
print(f'Ordem mínima do filtro Butterworth: K = {K}')

# Determinando a frequência de corte
Wc_p = W_p/((10**(a_p/10)-1)**(1/(2*K)))
Wc_s = W_s/((10**(a_s/10)-1)**(1/(2*K)))
Wc_m = (Wc_p + Wc_s)/2
Wc = Wc_m
print(f'Frequência de corte do filtro Butterworth: Wc = {Wc:.4f} rad/s')

# # Calcular os polos do Filtro
i = np.arange(1,K+1)
poles = 1j*Wc*np.exp(1j*np.pi/(2*K)*(2*i-1))

denominator = np.real(np.poly(poles))
numerator = denominator[-1]
system = ct.tf(numerator, denominator)
print("Função de Transferência do Filtro Butterworth:")
print(system)

Hp = numerator/np.polyval(denominator, 1j*W_p)
Hs = numerator/np.polyval(denominator, 1j*W_s)
print(f'Magnitude em Wp = {W_p} rad/s: |H(jWp)| = {20*np.log10(np.abs(Hp)):.2f} dB')
print(f'Magnitude em Ws = {W_s} rad/s: |H(jWs)| = {20*np.log10(np.abs(Hs)):.2f} dB')

w = np.arange(0, 50, 0.01)
H = numerator/np.polyval(denominator, 1j*w)

plt.figure(figsize=(6,3))
plt.plot(w, 20*np.log10(np.abs(H)), color='black')
plt.xlabel('$\Omega$ (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.xlim(0, 40)
plt.ylim(-35,1)
plt.grid()
plt.tight_layout()
plt.show()