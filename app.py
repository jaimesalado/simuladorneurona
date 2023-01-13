import streamlit as st
import numpy as np
import math

style = f"""
<style>
    .appview-container .main .block-container{{
        max-width: 90%;
    }}
</style>
"""
st.markdown(style, unsafe_allow_html=True)

# imagen
from PIL import Image
image = Image.open('neurona.jpg')
st.image(image)


# titulo
st.header('Simulador de neurona')


# funciones

def sigmoid(x):
  return 1/(1+np.exp(-x))

def relu(x):
  return np.maximum(0, x)

def tanh(x):
  return np.tanh(x)



# clase neuron

class Neuron:
  def __init__(self, weights, bias, func):
    self.weights = weights
    self.bias = bias
    self.func = func

  def change_bias(self, bias):
    self.bias = bias

  def choose_function(self, sigma):
    functions = {
        'relu' : relu(sigma),
        'tanh' : tanh(sigma),
        'sigmoid' : sigmoid(sigma)
    }
    return functions[self.func]

  def run(self, input_data):
    x = np.array(input_data)
    weigths = np.array(self.weights)
    sigma = np.dot(x, weigths) + (self.bias)
    return self.choose_function(sigma)

  @property
  def show_data(self):
    print("weights: ", self.weights)
    print("Bias: ", self.bias)
    print("Func: ", self.func)


# seleccionar numero de pesos y entradas

numero_entradas_pesos = st.slider("Selecciona el número de entradas y pesos de la neurona", 1, 10)


# seleccionar peso

st.subheader("Pesos")

pesos = []
numero_pesos = st.columns(numero_entradas_pesos)

for i in range(numero_entradas_pesos):
    pesos.append(i)
    with numero_pesos[i]:
        st.markdown(f"Peso {i}")
        pesos[i] = st.number_input(f"Peso {i}", label_visibility="collapsed")
st.text(f"El peso seleccionado es: {pesos}")




# seleccionar entradas

st.subheader("Entradas")

entradas = []
numero_entradas = st.columns(numero_entradas_pesos)

for i in range(numero_entradas_pesos):
    entradas.append(i)

    with numero_entradas[i]:
        st.markdown(f"Entrada {i}", unsafe_allow_html=True)
        entradas[i] = st.number_input(f"Entrada {i}", label_visibility="collapsed")

st.text(f"El peso seleccionado es: {entradas}")


# seleccionar sesgo

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sesgo")
    sesgo = st.number_input("Selecciona el valor del sesgo")
st.text(f"El sesgo seleccionado es:  {sesgo}")


with col2:
    st.subheader("Función de activación")
    elegir_funcion = st.selectbox('Elige la función de activación',('Sigmoide', 'ReLU', 'Tangente hiperbólica'))
st.text(f"La función seleccionada es:  {elegir_funcion}")



funciones = {'Sigmoide': 'sigmoid', 'ReLU': 'relu', 'Tangente hiperbólica': 'tanh'}
if st.button("Calcular salida de la neurona"):
    neurona = Neuron(weights=pesos, bias=sesgo, func=funciones[elegir_funcion])
    st.text(f"El resultado de la neurona es:  {neurona.run(input_data=entradas)}")
