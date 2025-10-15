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

#Funcao Pulso de Amostragem
def Pulse(t, tau):
    return 1*np.logical_and(t>=0, t<=tau)

#Parametros de Simulacao
Td = 1e-5
time = 1.0
t = np.arange(0,time,Td)

#Intervalo de amostragem
Fs = 50
Ts = 1/Fs

#Gerar o trem de pulsos
tau = 0.005
p = Pulse(np.mod(t,Ts), tau)

#Sinal a ser amostrado
g = np.sinc(5*(t-0.4))**2
f,Gf = ctft(g,Td)

#Sinal apos a amostragem
gT = g*p
f,GTf = ctft(gT,Td)

plt.figure(figsize=(10,8))
plt.subplot(4,1,1)
plt.plot(t,p,'k')
plt.xlabel('$t$')
plt.title('Pulso de amostragem')
plt.xlim(min(t),max(t))
plt.grid()

plt.subplot(4,1,2)
plt.plot(t,g,'--k')
plt.plot(t,gT,'-b')
plt.xlabel('$t$')
plt.title('Sinal original e suas amostras (prático)')
plt.xlim(min(t),max(t))
plt.grid()

plt.subplot(4,1,3)
plt.plot(f,np.abs(Gf),'k')
plt.xlabel('$f$')
plt.title('Espectro do sinal original')
plt.xlim(-150,150)
plt.grid()

plt.subplot(4,1,4)
plt.plot(f,np.abs(GTf),'k')
plt.xlabel('$f$')
plt.title('Espectro do sinal amostrado')
plt.xlim(-150,150)
plt.grid()

plt.tight_layout()
plt.show()