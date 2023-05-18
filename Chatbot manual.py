#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import numpy as np
import random
import string


# # 1 DEFINICIÓN CORPUS

# In[2]:


f=open(r'G:\Mi unidad\Universidad de Morón\Tesis de grado\Tesis\Procesamiento del Lenguaje Natural con Python (NLP)\Sección 5 - Chatbot\Corpus_crucero.txt','r',errors = 'ignore')
raw=f.read()


# In[3]:


import re
from unicodedata import normalize

# -> NFD y eliminar diacríticos
raw = re.sub(
        r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", 
        normalize( "NFD", raw), 0, re.I
    )

print(raw)


# # 2a Preprocesamiento del Texto con NTLK CORPUS

# In[4]:


raw=raw.lower()# convertir en minúscula
#nltk.download('punkt') # Instalar módulo punkt si no está ya instalado (solo ejecutar la primera vez)
#nltk.download('wordnet') # Instalar módulo wordnet si no está ya instalado (solo ejecutar la primera vez)
nltk.download('stopwords')
sent_tokens = nltk.sent_tokenize(raw)# Convierte el CORPUS a una lista de sentencias
word_tokens = nltk.word_tokenize(raw)# Convierte el CORPUS a una lista de palabras
lemmer = nltk.stem.WordNetLemmatizer()

#WordNet diccionario semántico incluido en NLTK
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# # 2b PREPROCESAMIENTO DEL TEXTO + 3 Evaluar Similitud MENSAJE USUARIO - CORPUS

# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

#Función para determinar la similitud del texto insertado y el corpus
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response) #Añade al corpus la respuesta de usuario al final
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=stopwords.words('spanish'))
    tfidf = TfidfVec.fit_transform(sent_tokens)
    # 3 EVALUAR SIMILITUD DE COSENO ENTRE MENSAJE USUARIO (tfidf[-1]) y el CORPUS (tfidf)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if(req_tfidf==0):
        robo_response=robo_response+"Lo siento, no te he entendido. Si no puedo responder a lo que busca póngase en contacto con el profesor correspondiente"
        return robo_response

    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


# # 4 DEFINICIÓN DE COINCIDENCIAS MANUAL

# In[6]:


SALUDOS_INPUTS = ("hola", "buenas", "saludos", "qué tal", "hey","buenos dias",)
SALUDOS_OUTPUTS = ["Hola", "Hola, ¿Qué tal?", "Hola, ¿Cómo te puedo ayudar?", "Hola, encantado de hablar contigo"]

def saludos(sentence):
    for word in sentence.split():
        if word.lower() in SALUDOS_INPUTS:
            return random.choice(SALUDOS_OUTPUTS)


# # 5 GENERACIÓN DE RESPUESTA

# In[ ]:


flag=True
print("ROBOT: Mi nombre es ROBOT. Contestaré a tus preguntas acerca de sus vacaciones en el crucero. Si quieres salir, escribe 'salir' ")
while(flag==True):
    user_response = input()
    user_response = user_response.lower() #Convertimos a minúscula
    
    # -> NFD y eliminar diacríticos
    user_response = re.sub(
        r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", 
        normalize( "NFD", user_response), 0, re.I
    )
    
    if(user_response!='salir'):
        
        if(user_response=='gracias' or user_response=='muchas gracias'): #Se podría haber definido otra función de coincidencia manual
            flag=True
            print("ROBOT: No hay de qué")
            
        else:
            if(saludos(user_response)!=None): #Si la palabra insertada por el usuario es un saludo (Coincidencias manuales definidas previamente)
                print("ROBOT: "+saludos(user_response))
                
            else: #Si la palabra insertada no es un saludo --> CORPUS
                print("ROBOT: ",end="") 
                print(response(user_response))
                sent_tokens.remove(user_response) # para eliminar del corpus la respuesta del usuario y volver a evaluar con el CORPUS limpio
    else:
        flag=False
        print("ROBOT: Nos vemos pronto, ¡cuídate!")


# In[ ]:




