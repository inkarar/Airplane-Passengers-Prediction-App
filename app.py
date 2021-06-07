
# LSTM for international airline passengers problem with regression framing
import streamlit as st
import numpy
import matplotlib.pyplot as plt
import plotly.express as px
from pandas import read_csv
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

st.write("""
# Airplane Passengers Prediction App
""")

# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')
st.sidebar.write('Play around with these parameters')

look_back = st.sidebar.slider('Window Size in No of Months ---->', 1, 20, 6)
epochs = st.sidebar.slider('No of Epochs/Iteraions of Dataset ---->', 1, 50,19)
train_size = st.sidebar.slider('Train Dataset Size(%) ---->', 1, 100,64)

df = pd.DataFrame({'look_back_terms':look_back,'epochs':epochs,"train_dataset_size":train_size},index=[0])

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
##uploaded_file = st.file_uploader("Please Upload AirPassengers.csv", type="csv")
##
##if uploaded_file:
##        dataframe = read_csv(uploaded_file, usecols=[1], engine='python')

dataframe = read_csv("AirPassengers.csv", usecols=[1], engine='python')    
dataset = dataframe.values
dataset = dataset.astype('float32')


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)



# split into train and test sets
#train_size = int(len(dataset) * 0.67)
train_size = int(len(dataset) * train_size * 0.01)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


# reshape into X=t and Y=t+1
trainX, trainY = create_dataset(train)
testX, testY = create_dataset(test)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)       

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
#print('Train Score: %.2f RMSE' % (trainScore))

st.sidebar.write("--------------")

st.sidebar.write("""
# Model Results
model error - smaller the error, the better
""")

st.sidebar.write('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#print('Test Score: %.2f RMSE' % (testScore))
st.sidebar.write('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

#plots
fig1 = px.line(x=dataframe.index, y=dataframe['#Passengers'], labels={'x':"Time in Months", 'y':"No of Passengers"})
st.write("""
## Original Dataset (1999 - 2010)
""")
st.plotly_chart(fig1, use_container_width=True, sharing='streamlit')

st.write("""
## Train-Test Dataset Split 
""")
st.area_chart(numpy.append(trainPredictPlot,testPredictPlot,axis=1),use_container_width=True)

st.write("""
## Monthly-Yearly Trend in Dataset (1999 - 2010)
""")
st.image('monthlytrend.png',width=600)
st.write('Evidently, JULY and AUGUST are the crowdiest months!')


st.write("""
## FORECASTING No of Airplane Passengers
""")
st.line_chart(numpy.append(trainPredictPlot,testPredictPlot,axis=1),use_container_width=True)


st.write("------------------------------")

col1, col2 = st.beta_columns(2)

col1.image('Forecasting Window.png')

col2.image('Prediction.png')

future = read_csv("AirPassengers - future.csv", usecols=[1], engine='python')
st.write(future.iloc[144])
future = future.iloc[144:]
fig2 = px.line(x=future.index, y=future['#Passengers'], labels={'x':"Time in Months", 'y':"No of Passengers"})
st.write("""
## Forecasting 24 months into the Future (2011 - 2012)
""")
st.plotly_chart(fig2, use_container_width=True, sharing='streamlit')
