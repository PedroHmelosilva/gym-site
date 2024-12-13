import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Carregar o dataset
df = pd.read_csv('gym_members_exercise_tracking.csv')

# Selecionar colunas de interesse
columns = ['Weight', 'Height', 'Resting_BPM', 'Session_Duration', 'Avg_BPM']
df = df[columns]

# Dividir entre variáveis independentes (X) e dependente (y)
X = df.iloc[:, 0:4]  # Pesos, Altura, BPM fora do treino, Duração
y = df['Avg_BPM']    # BPM médio durante o treino (coluna como série unidimensional)

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Salvar o modelo treinado
pickle.dump(model, open('model.pkl', 'wb'))