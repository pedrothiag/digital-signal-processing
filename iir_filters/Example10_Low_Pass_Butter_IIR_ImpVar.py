import numpy as np
import matplotlib.pyplot as plt
import control as ct
np.set_printoptions(suppress=True)

#Muda a fonte dos Plots para Times New Roman
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'
plt.rcParams['font.family'] = 'lmodern'

K = 6
Omega_c = 70.30205
i = np.arange(1,K+1)
poles = 1j*Omega_c * np.exp(1j*np.pi/(2*K)*(2*i -1))

denominator = np.real(np.poly(poles))
numerator = denominator[-1]
print(f'denominator: {denominator}')
print(f'numerator: {numerator}')


sysc = ct.tf(numerator, denominator)
sysd_impulse_invariant = ct.sample_system(sysc, 1e-2, method='impulse')
#print(sysd_impulse_invariant)

num_d = sysd_impulse_invariant.num[0][0]
den_d = sysd_impulse_invariant.den[0][0]

w = np.arange(0, np.pi, 0.01)
H = np.polyval(num_d, np.exp(-1j*w)) / np.polyval(den_d, np.exp(-1j*w))

w0 = 0.3*np.pi
H0 = 20*np.log10(np.abs(np.polyval(num_d, np.exp(-1j*w0)) / np.polyval(den_d, np.exp(-1j*w0))))
print(H0)

Ts = 1e-2
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
    xytext=(w0 + 5, H0 + 5),# posição do texto
    arrowprops=dict(arrowstyle='->', color='k', lw=1),
    color='k',
    fontsize=10
)
plt.grid()
plt.tight_layout()
plt.show()