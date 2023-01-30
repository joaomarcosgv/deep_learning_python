#AGRUPAMENTO
#Algoritmo de Lloyd (k-means)
#Usado para mapas auto-organizáveis
#k é o número de centroides, sendo um centroide para cada cluster
#Distância euclidiana é a raiz do somatorio dos quadrados das diferenças entre as coordenadas
#O centroide é o ponto em que cada coordenada é a médias das coordenadas dos pontos do grupo
#Uma boa estimativa do número de clusters é a raiz da metade do número de registros da base
#O número de clusters é empírico e arbitrado, porém o Elbow method otimiza o número com base no somatório dos quadrados

#Mapas auto-organizáveis (self organizing maps - SOM)
#Redes neurais artificiais (ANN) em que a camada de saída é o próprio mapa
#Eles não tem camada escondida nem função de ativação
#Tamanho do SOM é cinco vezes a raíz do número de registros
#Cada ponto do mapa tem um conjunto de pesos, um por entrada
#Esses pesos são inicializados aleatoriamente e cada registro de entrada é atribuído a uma das saídas, a qual é chamada de sua BMU (best matching unit)


from minisom import MiniSom
import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')
base = base.dropna()
base.loc[base.age < 0, 'age'] = 40.92

x = base.iloc[:, 0:4].values
y = base.iloc[:, 4].values

#Normalizar valores de 0 a 1
from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range = (0,1))
x = normalizador.fit_transform(x)

#Construção do mapa
som = MiniSom(x = 15, y = 15, input_len = 4, random_seed = 0) 
#Inicialização dos pesos
som.random_weights_init(x)
#Treinamento
som.train_random(data = x, num_iteration = 100)

som._weights
som._activation_map
q = som.activation_response(x)

#MID (mean inter neuron distance) varia de 0 a 1
#0 significa que a distancia euclidiana para os vizinhos é grande
from pylab import pcolor, colorbar, plot
pcolor(som.distance_map().T)
colorbar()

#Acha o neurônio ganhador (BMU) de cada um dos registros
#O marcador e sua cor serve apenas para indicar qual das três classes
markers = ['o', 's']
color = ['r', 'g']
for i, x in enumerate(x):
     w = som.winner(x) #Acha o BMU
     plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
          markerfacecolor = 'None', markersize = 10,
          markeredgecolor = color[y[i]], markeredgewidth = 2)

#Indica quais registros estão associados a cada um dos neurônios     
mapeamento = som.win_map(x) 
suspeitos = np.concatenate(mapeamento[(13,9)], mapeamento[(1,10)], axis = 0)
#Faz a operação inversa da normalização
suspeitos = normalizador.inverse_transform(suspeitos)

classe = []
for i in range(len(suspeitos)):
    for j in range(len(suspeitos)):
        if base.iloc[i,0] == int(round(suspeitos[j,0])):
            classe.append(base.iloc[i,4])
classe = np.asarray(classe)

suspeitos_final = np.column_stack((suspeitos, classe))
suspeitos_final = suspeitos_final[suspeitos_final[:, 4].argsort()]