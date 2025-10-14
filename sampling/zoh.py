import numpy as np
import numpy.fft as fft
import scipy.signal as signal
import matplotlib.pyplot as plt

#Muda a fonte dos Plots para Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams['mathtext.rm'] = "Times New Roman"
plt.rcParams['mathtext.it'] = "Times New Roman:italic"
plt.rcParams['mathtext.bf'] = "Times New Roman:bold"

#Funcao para Calcular a Transformada de Fourier
def ctft(x,Ts):
    Fs = 1/Ts
    N = len(x)
    f = np.linspace(-Fs/2,Fs/2,N)
    Xf = fft.fftshift(fft.fft(Ts*x))
    return f,Xf

# Parâmetros de Simulação
Td = 1e-4
t = np.arange(0,1.0,Td)

#Retentor de Ordem Zero
Fs = 50
Ts = 1/Fs
nFac = int(np.round(Ts/Td))
hZOH = np.zeros_like(t)
hZOH[0:nFac] = 1
f, Hf_zoh = ctft(hZOH,Td)

plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t,hZOH,'k')
plt.xlabel('$t$')
plt.title('Resposta ao impulso do retentor de ordem zero')
plt.xlim(0,5*Ts)
plt.grid()

plt.subplot(2,1,2)
plt.plot(f,np.abs(Hf_zoh),'k')
plt.xlabel('$f$')
plt.title('Resposta em frequência do retentor de ordem zero')
plt.xlim(-200,200)
plt.grid()

plt.tight_layout()

###############################################################################################
# Amostragem Ideal
###############################################################################################
# Funcao g = sinc(5*pi*(t - 0.4))^2 e sua transformada de Fourier
g = np.sinc(5*(t-0.4))**2
f,Gf = ctft(g,Td)

# Amostragem de g(t)
N = len(g)
nFac = int(np.round(Ts/Td))         # A cada nFac valores de g(t) teremos uma amostra de gT(t)
gT = np.zeros(N)
gT[0::nFac] = g[0::nFac]
f,GTf = ctft(gT,Td)

gT = np.zeros_like(g)
gT[0::nFac] = g[0::nFac]
f,GTf = ctft(gT,Td)

###############################################################################################
# Reconstrução por ZOH
###############################################################################################
g_zoh_temp = np.convolve(gT,hZOH)
Nconv = len(gT)
g_zoh = np.zeros(Nconv)
g_zoh = g_zoh_temp[0:Nconv]
f,G_zoh_f = ctft(g_zoh,Td)

plt.figure(figsize=(10,8))
plt.subplot(4,1,1)
plt.plot(t,g,'--k')
plt.xlabel('$t$')
plt.title('Sinal $g(t)$ e a reconstrução de amostras por ZOH')
plt.plot(t,g_zoh,'b')
plt.xlim(min(t),max(t))
plt.grid()

plt.subplot(4,1,2)
plt.plot(f,abs(Gf),'k')
plt.xlim(-150,150)
plt.xlabel('$f$')
plt.title('Espectro de $g(t)$')
plt.grid()

plt.subplot(4,1,3)
plt.plot(f,abs(GTf),'k')
plt.xlim(-150,150)
plt.xlabel('$f$')
plt.title('Espectro de $g_T(t)$')
plt.grid()

plt.subplot(4,1,4)
plt.plot(f,abs(G_zoh_f),'k')
plt.xlim(-150,150)
plt.xlabel('$f$')
plt.title('Espectro de $g_{ZOH}(t)$')
plt.grid()

plt.tight_layout()

#Filtro Passa-Baixas de Equalizacao
fc = Fs/2                                                # frequência de corte
ordem = 4                                               # ordem do filtro
Fd = 1/Td
Wn = fc / (Fd/2)                                    # frequência de corte normalizada
b_lp, a_lp = signal.butter(ordem, Wn, btype='lowpass')   # cálculo dos coeficientes do filtro

g_rec_zoh = signal.filtfilt(b_lp, a_lp, g_zoh)
f,G_rec_zoh = ctft(g_rec_zoh,Td)

plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t,g,'k', label='Sinal original')
plt.plot(t,g_rec_zoh,'--b', label='Sinal reconstruído')
plt.xlabel('$t$')
plt.title('Sinal original vs. sinal reconstruído')
plt.xlim(min(t),max(t))
plt.legend(loc=1)
plt.grid()

plt.subplot(2,1,2)
plt.plot(f,abs(G_rec_zoh),'k')
plt.xlim(-150,150)
plt.xlabel('$f$')
plt.title('Espectro do sinal reconstruído')
plt.tight_layout()
plt.grid()

plt.show()