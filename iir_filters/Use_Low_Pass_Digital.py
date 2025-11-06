import numpy as np
import matplotlib.pyplot as plt

"""
  0.00936 z^2 + 0.01872 z + 0.00936
  --------------------------------- = Y(z) / X(z)
       z^2 - 1.771 z + 0.8178
  y[n+2] - 1.771 y[n+1] + 0.8178 y[n] = 0.00936 x[n+2] + 0.01872 x[n+1] + 0.00936 x[n]
  y[n] = 1.771 y[n-1] - 0.8178 y[n-2] + 0.00936 x[n] + 0.01872 x[n-1] + 0.00936 x[n-2]
"""

# Criar o sinal de entrada a ser filtrado
Ts = 0.01  # Período de amostragem
w1 = 3*np.pi
w2 = 25*np.pi
tend = 3
t = np.arange(0, tend, Ts)
#x = np.cos(w1*t) + np.sin(w2*t)
x = np.cos(w1*t) + np.random.randn(len(t))*0.5  # Sinal com Ruído

# Sistema em Repouso
yn1 = 0.0  # y[n-1]
yn2 = 0.0  # y[n-2]
xn1 = 0.0  # x[n-1]
xn2 = 0.0  # x[n-2]

N = len(x)
y = np.zeros(N)
for i in range(N):
    xn = x[i]           # xn = Ler_ADC()
    yn = 1.771*yn1 - 0.8178*yn2 + 0.00936*xn + 0.01872*xn1 + 0.00936*xn2
    y[i] = yn           # yn = Escrever_DAC(yn)
    
    # Atualiza os valores para a próxima iteração
    yn2 = yn1
    yn1 = yn
    xn2 = xn1
    xn1 = xn

plt.figure(figsize=(8,4))
plt.subplot(2,1,1)
plt.plot(t, x, color='blue')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Sinal de Entrada x[n]')
plt.grid()
plt.subplot(2,1,2)
plt.plot(t, y, color='red')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Sinal de Saída y[n] - Após Filtro IIR Passa-Baixas')
plt.grid()
plt.tight_layout()
plt.show()