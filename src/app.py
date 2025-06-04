import re                       # Para trabajar con expresiones regulares (usado para separar URLs).
import nltk                    # Toolkit para procesamiento de lenguaje natural.
from sklearn.model_selection import train_test_split   # Para dividir los datos en train y test.
from sklearn.feature_extraction.text import TfidfVectorizer  # Para convertir texto en vectores numéricos.
from nltk.corpus import stopwords       # Para obtener palabras vacías ("stopwords") del inglés.
from nltk.stem import WordNetLemmatizer  # Para lematizar palabras (obtener la forma base).

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv"
df = pd.read_csv(url)

data = pd.DataFrame(df)

data

data["label"] = data["is_spam"].astype(int)
data

print(data.shape)
print(f"Spam: {len(data.loc[data.label == 1])}")
print(f"No spam: {len(data.loc[data.label == 0])}")

data = data.drop_duplicates()
data = data.reset_index(inplace = False, drop = True)
data.shape

import regex as re

def preprocess_text(text):
    # Eliminar cualquier caracter que no sea una letra (a-z) o un espacio en blanco ( )
    text = re.sub(r'[^a-z ]', " ", text)
    
    # Eliminar espacios en blanco
    text = re.sub(r'\s+[a-zA-Z]\s+', " ", text)
    text = re.sub(r'\^[a-zA-Z]\s+', " ", text)

    # Reducir espacios en blanco múltiples a uno único
    text = re.sub(r'\s+', " ", text.lower())

    # Eliminar tags
    text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", text)

    return text.split()

data["url"] = data["url"].apply(preprocess_text)
data.head()

from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
download("wordnet")
lemmatizer = WordNetLemmatizer()

download("stopwords")
stop_words = stopwords.words("english")

def lemmatize_text(words, lemmatizer = lemmatizer):
    tokens = [lemmatizer.lemmatize(word) for word in words]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 3]
    return tokens

data["url"] = data["url"].apply(lemmatize_text)
data.head()

import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud(width = 800, height = 800, background_color = "black", max_words = 1000, min_font_size = 20, random_state = 42)\
    .generate(str(data["url"]))

fig = plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

from sklearn.model_selection import train_test_split #

x_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.svm import SVC

model = SVC(kernel = "linear", random_state = 42)
model.fit(x_train, y_train)

y_pred = model.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(conf_matrix)

# Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# F1 Score
f1 = f1_score(y_test, y_pred)
print("\nF1 Score:", f1)

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(x_train, y_train)

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Definir el espacio de búsqueda
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'class_weight': [None, 'balanced']
}

# Inicializar el modelo base
svc = SVC()

# Grid Search
grid_search = GridSearchCV(estimator=svc,
                           param_grid=param_grid,
                           cv=5,
                           scoring='f1',
                           verbose=2,
                           n_jobs=-1)

# Entrenar el modelo
grid_search.fit(x_train, y_train)

print("Mejores parámetros encontrados:")
print(grid_search.best_params_)

print("\nMejor F1 Score en cross-validation:")
print(grid_search.best_score_)

# Mejor modelo
best_model = grid_search.best_estimator_

# Predicciones
y_pred = best_model.predict(X_test)

# Métricas
from sklearn.metrics import confusion_matrix, classification_report, f1_score

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

print("F1 Score:", f1_score(y_test, y_pred))

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import pandas as pd

# 1. Carga de datos
url = "https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv"
data = pd.read_csv(url)

# 2. Asignar X e y
X = data["url"]
y = data["is_spam"].astype(int)  # Asegúrate que sea int (0 o 1)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Pipeline con SMOTE
pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, max_df=0.8, min_df=5)),
    ('smote', SMOTE(random_state=42)),
    ('svc', SVC())
])

# 5. Grid de hiperparámetros
param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf'],
    'svc__class_weight': ['balanced'],
    'svc__gamma': ['scale', 'auto']
}

# 6. GridSearchCV
grid_search = GridSearchCV(
    pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=2
)
grid_search.fit(X_train, y_train)

# 7. Resultados
print("\nMejores parámetros encontrados:")
print(grid_search.best_params_)
print("\nMejor F1 Score en cross-validation:")
print(grid_search.best_score_)

# 8. Evaluación en test
y_pred = grid_search.predict(X_test)

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

print("F1 Score:", f1_score(y_test, y_pred))

import joblib

# Guardar modelo
joblib.dump(grid_search.best_estimator_, "modelo_svm_optimizado.joblib")