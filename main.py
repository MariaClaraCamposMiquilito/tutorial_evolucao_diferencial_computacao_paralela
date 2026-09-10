# Bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba


###### DADOS EXPERIMENTAIS #######

# Lendo os Dados dos .csv
posicoes_y = pd.read_csv("https://raw.githubusercontent.com/MariaClaraCamposMiquilito/tutorial_evolucao_diferencial_computacao_paralela/refs/heads/main/posicoes_y.csv")
posicoes_x = pd.read_csv("https://raw.githubusercontent.com/MariaClaraCamposMiquilito/tutorial_evolucao_diferencial_computacao_paralela/refs/heads/main/posicoes_x.csv")
velocidades = pd.read_csv("https://raw.githubusercontent.com/MariaClaraCamposMiquilito/tutorial_evolucao_diferencial_computacao_paralela/refs/heads/main/velocidades.csv")
posicoes = pd.read_csv("")

# Extraindo somente os componenentes das velocidades
velocidade_x = velocidades[['t', 'vx']]
velocidade_y = velocidades[['t','vy']]

