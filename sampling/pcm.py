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

#Funcao para fazer a quantizacao
def uniquan(x, L):
    #Primeiramente define-se o valor de Delta
    x_max = max(x)
    x_min = min(x)
    Delta = (x_max - x_min)/L

    #Cria-se um vetor com os niveis de quantizacao. Eles estão igualmente distribuidos. 
    #O deslocamento por Delta/2 é devido a aproximacao pelo valor medio
    q_level = np.linspace(x_min + Delta/2, x_max - Delta/2, L)

    #Ao inves de percorrer o sinal e ver qual o valor de quantizacao ele mais se aproxima
    #o que fazemos aqui é mapear o sinal para valores entre 0 e L-1 e arredonda esses valor
    a_p = (L-1)/(x_max - x_min)
    b_p = -a_p*x_min
    x_p = np.round((a_p*x + b_p)).astype(int)

    #Sinal codificado
    x_pcm = q_level[x_p]
    return x_pcm
    
Td = 1e-5
time = 1.0
t = np.arange(0,time,Td)

#Retentor de Ordem Zero
Fs = 50
Ts = 1/Fs
nFac = int(np.round(Ts/Td))
hZOH = np.zeros_like(t)
hZOH[0:nFac] = 1

g = np.sinc(5*(t-0.4))**2

gT = np.zeros_like(g)
gT[0::nFac] = g[0::nFac]

g_zoh_temp = np.convolve(gT,hZOH)
Nconv = int(len(g_zoh_temp)/2) + 1
g_zoh = np.zeros(Nconv)
g_zoh = g_zoh_temp[0:Nconv]

g_pcm_16 = uniquan(g_zoh, 16)
g_pcm_8  = uniquan(g_zoh, 8)
g_pcm_4  = uniquan(g_zoh, 4)

plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.plot(t,g,'--k')
plt.xlabel('$t$')
plt.title('Sinal $g(t)$ e o correspondente sinal PCM com 16 níveis')
plt.plot(t,g_pcm_16,'-b')
plt.xlim(min(t),max(t))
plt.grid()

plt.subplot(3,1,2)
plt.plot(t,g,'--k')
plt.xlabel('$t$')
plt.title('Sinal $g(t)$ e o correspondente sinal PCM com 8 níveis')
plt.plot(t,g_pcm_8,'-b')
plt.xlim(min(t),max(t))
plt.grid()

plt.subplot(3,1,3)
plt.plot(t,g,'--k')
plt.xlabel('$t$')
plt.title('Sinal $g(t)$ e o correspondente sinal PCM com 4 níveis')
plt.plot(t,g_pcm_4,'-b')
plt.xlim(min(t),max(t))
plt.grid()

plt.tight_layout()

#Filtro Passa-Baixas
W = Fs/2                                                # frequência de corte
ordem = 4                                               # ordem do filtro
Fd = 1/Td
fc_norm = W / (Fd/2)                                    # frequência de corte normalizada
b_lp, a_lp = signal.butter(ordem, fc_norm, btype='lowpass')   # cálculo dos coeficientes do filtro

g_rec_pcm_16 = signal.filtfilt(b_lp, a_lp, g_pcm_16)
g_rec_pcm_8 = signal.filtfilt(b_lp, a_lp, g_pcm_8)
g_rec_pcm_4 = signal.filtfilt(b_lp, a_lp, g_pcm_4)

plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.plot(t,g,'k', label='Sinal original')
plt.xlabel('$t$')
plt.title('Sinal $g(t)$ e o sinal reconstruído com PCM com 16 níveis')
plt.plot(t,g_rec_pcm_16,'--b', label='Sinal reconstruído')
plt.xlim(min(t),max(t))
plt.legend(loc=1)
plt.grid()

plt.subplot(3,1,2)
plt.plot(t,g,'k', label='Sinal original')
plt.xlabel('$t$')
plt.title('Sinal $g(t)$ e o sinal reconstruído com PCM com 8 níveis')
plt.plot(t,g_rec_pcm_8,'--b', label='Sinal reconstruído')
plt.xlim(min(t),max(t))
plt.legend(loc=1)
plt.grid()

plt.subplot(3,1,3)
plt.plot(t,g,'k', label='Sinal original')
plt.xlabel('$t$')
plt.title('Sinal $g(t)$ e o sinal reconstruído com PCM com 4 níveis')
plt.plot(t,g_rec_pcm_4,'--b', label='Sinal reconstruído')
plt.xlim(min(t),max(t))
plt.legend(loc=1)
plt.grid()

plt.tight_layout()
plt.show()