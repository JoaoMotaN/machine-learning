import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import seaborn as sns
from sklearn.decomposition import PCA

#seleção das colunas e tratramento dos dados
filme = pd.read_csv('movies_metadata.csv', low_memory = False)
filme = filme [['id','original_title','original_language','vote_count','genres','production_companies']]
filme.rename(columns = {'id':'ID_FILME','original_title':'TITULO','original_language':'LINGUAGEM','vote_count':'QT_AVALIACOES',
'genres':"GENEROS",'production_companies':'PRODUTORAS'}, inplace = True)
filme.dropna(inplace = True)
filme = filme[filme['QT_AVALIACOES'] > 999]
filme = filme[filme['LINGUAGEM'] == 'en']
filme['ID_FILME'] = filme['ID_FILME'].astype(int)

def get_genres_names(data: str) -> list:
    if isinstance(data, str):
        data = ast.literal_eval(data)
        names = [item['name'] for item in data]
        return names
    return []

def get_names(data: str) -> list:
    try:
        companies = ast.literal_eval(data)
        names = [company['name'] for company in companies]
        return names
    except (ValueError, TypeError):
        return []

filme['PRODUTORAS'] = filme['PRODUTORAS'].apply(get_names)
filme['GENEROS'] = filme['GENEROS'].apply(get_genres_names)

#juntando as colunas que serão usadas na criação do modelo de machine learning
filme['Infos'] = filme['GENEROS'] + (filme['PRODUTORAS'])

#uso do TF-IDF para transformação de uma matriz numerica
vec = TfidfVectorizer()
tfidf = vec.fit_transform(filme['Infos'].apply(lambda x: np.str_(x)))

#usando a similaridade do cosseno na matriz criada
sim = cosine_similarity(tfidf)

# transformando em um dataframe
sim_df2 = pd.DataFrame(sim, columns=filme['TITULO'], index=filme['TITULO'])
sim_df2.head()

st.write("Gráfico de Dispersão")

pca = PCA(n_components=2)
pca_result = pca.fit_transform(sim)
#st.scatter_chart(pca_result)
plt.figure(figsize=(12, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.title('Gráfico de Dispersão da Similaridade entre Filmes')
st.pyplot(plt)



