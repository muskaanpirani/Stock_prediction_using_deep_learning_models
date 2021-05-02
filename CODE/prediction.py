import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
from plotly import graph_objs as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import base64


st.title('Stock Prediction Using Deep Learning Model')
st.markdown("<hr>",
    unsafe_allow_html=True)

dataset = ('IBM',
           'TCS',
           'NSE',
           'SNOWMAN')


st.sidebar.markdown("<hr>",
    unsafe_allow_html=True)

option = st.sidebar.selectbox('Select Stock for Prediction',dataset)
DATA_URL =('./DB/'+option+'.csv')
year = st.sidebar.slider('Year of Prediction:',1,4)
period = year * 365

st.sidebar.markdown("<hr>",
    unsafe_allow_html=True)


@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    return data

st.title(option)

data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text('Loading data... done!')

st.markdown("<hr>",
    unsafe_allow_html=True)


st.subheader('Raw Data')
st.write(data)


st.markdown("<hr>",
    unsafe_allow_html=True)


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    st.subheader('Time Series data with Rangeslider')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()


st.markdown("<hr>",
    unsafe_allow_html=True)

data_pred = data[['Date','Close']]
data_pred = data_pred.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(data_pred)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

fig1 = plot_plotly(m, forecast)
st.subheader('Forecast Data')
st.write(forecast)


st.markdown("<hr>",
    unsafe_allow_html=True)

st.subheader('Forecasting closing of stock value for '+option+' for a period of: '+str(year)+' year')
st.plotly_chart(fig1)


st.markdown("<hr>",
    unsafe_allow_html=True)

st.subheader("Component wise forecast")
fig2 = m.plot_components(forecast)
st.write(fig2)

st.markdown("<hr>",
    unsafe_allow_html=True)


