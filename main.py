# Bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba
import math
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

###### DADOS EXPERIMENTAIS #######

# Lendo os Dados dos .csv
velocidades = pd.read_csv("https://raw.githubusercontent.com/MariaClaraCamposMiquilito/tutorial_evolucao_diferencial_computacao_paralela/refs/heads/main/Dados_Experimentais/velocidades.csv")
posicoes = pd.read_csv("https://raw.githubusercontent.com/MariaClaraCamposMiquilito/tutorial_evolucao_diferencial_computacao_paralela/refs/heads/main/Dados_Experimentais/posicoes.csv")


####### PARÂMETROS #######
gamma = 0.5
g     = 9.81 # m/s²

####### CONDIÇÕES INICIAIS #######
y0  = posicoes['y'].iloc[0]     # m
x0  = posicoes['x'].iloc[0]     # m
vy0 = velocidades['vy'].iloc[0] # m/s
vx0 = velocidades['vx'].iloc[0] # m/s

estado_inicial = [y0, x0, vy0, vx0]

t0 = 0.0
tf = 0.53
t = np.arange(t0, tf, 0.005)
t_span = (t0, tf)

####### MODELO #######
def modelo(t, y, gamma, g):
    yp, xp, vy, vx = y

    dypdt = vy
    dxpdt = vx
    dvydt = - g - (gamma * math.sqrt((vx ** 2) + (vy **2)) * vy) 
    dvxdt = - (gamma * math.sqrt((vx ** 2) + (vy **2)) * vx)

    return [dypdt, dxpdt, dvydt, dvxdt]

####### FUNÇÃO OBJETIVO #######
def funcao_objetivo(gamma):

    sol = solve_ivp(modelo, t_span, estado_inicial, args = (gamma, g), method = 'Radau', t_eval = t)
    
    # Interpolando os passos de tempo
    y_interp = np.interp(posicoes['t'], t, sol.y[0])
    x_interp = np.interp(posicoes['t'], t, sol.y[1])
    vy_interp = np.interp(velocidades['t'], t, sol.y[2])
    vx_interp = np.interp(velocidades['t'], t, sol.y[3])

    # Clip do modelo
    y_interp = np.clip(y_interp, 1e-12, None)
    x_interp = np.clip(x_interp, 1e-12, None)
    vy_interp = np.clip(vy_interp, 1e-12, None)
    vy_interp = np.clip(vx_interp, 1e-12, None)

    # Log10 do modelo
    y_sim  = np.log10(y_interp)
    x_sim  = np.log10(x_interp)
    vy_sim = np.log10(vy_interp)
    vx_sim = np.log10(vx_interp)
    # Log10 dos dados
    y_obs  = np.log10(posicoes['y'])
    x_obs  = np.log10(posicoes['x'])
    vy_obs = np.log10(velocidades['vy'])
    vx_obs = np.log10(velocidades['vx'])

    erro_y = (y_obs - y_sim) ** 2
    erro_x = (x_obs - x_sim) ** 2
    erro_vy = (vy_obs - vy_sim) ** 2
    erro_vx = (vx_obs - vx_sim) ** 2

    return np.sum(erro_y) + np.sum(erro_x) + np.sum(erro_vy) + np.sum(erro_vx)

if __name__ == '__main__':
    bounds = [(0, 1)]

    result = differential_evolution(
            funcao_objetivo, 
            bounds, 
            strategy = 'best1bin',
            popsize = 50,
            mutation = (0.5, 1),
            recombination = 0.7,
            disp = True)

####### GRÁFICOS ####### 
fig,ax = plt.subplots(2, 3, figsize = (18, 10))

## Posição Y em função do tempo
ax[0,0].plot(sol.t, sol.y[0], color = 'red', label = "Modelo")
ax[0,0].scatter(posicoes['t'], posicoes['y'], color = 'black', label = 'Dados experimentais')
ax[0,0].set_title('Variação da Posição Y da bolinha em função do tempo')
ax[0,0].set_xlabel('Tempo (s)')
ax[0,0].set_ylabel('Posição (m)')
ax[0,0].grid()
ax[0,0].legend()

## Posição X em função do tempo
ax[0,1].plot(sol.t, sol.y[1], color = 'blue', label = "Modelo")
ax[0,1].scatter(posicoes['t'], posicoes['x'], color = 'black', label = 'Dados experimentais')
ax[0,1].set_title('Variação da Posição X da bolinha em função do tempo')
ax[0,1].set_xlabel('Tempo (s)')
ax[0,1].set_ylabel('Posição (m)')
ax[0,1].grid()
ax[0,1].legend()

## Velocidade Y em função do tempo
ax[0,2].plot(sol.t, sol.y[2], color = 'green', label = "Modelo")
ax[0,2].scatter(velocidades['t'], velocidades['vy'], color = 'black', label = 'Dados experimentais')
ax[0,2].set_title('Variação da velocidade Y da bolinha em função do tempo')
ax[0,2].set_xlabel('Tempo (s)')
ax[0,2].set_ylabel('Velocidade (m/s)')
ax[0,2].grid()
ax[0,2].legend()

## Velocidade X em função do tempo
ax[1,0].plot(sol.t, sol.y[3], color = 'yellow', label = "Modelo")
ax[1,0].scatter(velocidades['t'], velocidades['vx'], color = 'black', label = 'Dados experimentais')
ax[1,0].set_title('Variação da velocidade X da bolinha em função do tempo')
ax[1,0].set_xlabel('Tempo (s)')
ax[1,0].set_ylabel('Velocidade (m/s)')
ax[1,0].grid()
ax[1,0].legend()

## Posição X versus Posição Y
ax[1,1].plot(sol.y[1], sol.y[0], color = 'brown', label = "Modelo")
ax[1,1].scatter(posicoes['x'], posicoes['y'], color = 'black', label = 'Dados experimentais')
ax[1,1].set_title('Trajetória da bolinha no lançamento')
ax[1,1].set_xlabel('Posição X (m)')
ax[1,1].set_ylabel('Posição Y (m)')
ax[1,1].grid()
ax[1,1].legend()

ax[1,2].axis('off')

plt.show()