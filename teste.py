# Bibliotecas
import numpy as np
import pandas as pd

# 1. Leitura dos Dados
url_base = "https://raw.githubusercontent.com/MariaClaraCamposMiquilito/tutorial_evolucao_diferencial_computacao_paralela/refs/heads/main/Dados_Experimentais/"
posicoes = pd.read_csv(url_base + "posicoes.csv")
velocidades = pd.read_csv(url_base + "velocidades.csv")

# 2. Extração dos arrays originais
t_pos = posicoes['t'].values
x_orig = posicoes['x'].values
y_orig = posicoes['y'].values

t_vel = velocidades['t'].values
vx_orig = velocidades['vx'].values
vy_orig = velocidades['vy'].values

# 3. Intervalo temporal comum (interseção)
t_min = max(t_pos.min(), t_vel.min())
t_max = min(t_pos.max(), t_vel.max())

# Grade temporal unificada
n_pontos = len(t_pos)
t_geral = np.linspace(t_min, t_max, n_pontos)

# 4. Interpolação correta de cada grandeza
x_interp  = np.interp(t_geral, t_pos, x_orig)
y_interp  = np.interp(t_geral, t_pos, y_orig)
vx_interp = np.interp(t_geral, t_vel, vx_orig)
vy_interp = np.interp(t_geral, t_vel, vy_orig)

# 5. DataFrame consolidado e exportação
dados_sincronizados = pd.DataFrame({
    "t": t_geral,
    "x": x_interp,
    "y": y_interp,
    "vx": vx_interp,
    "vy": vy_interp
})

dados_sincronizados.to_csv('dados_sincronizados.csv', index=False, encoding='utf-8')

print(dados_sincronizados.head())