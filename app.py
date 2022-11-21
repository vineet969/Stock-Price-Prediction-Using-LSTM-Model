import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model 
import streamlit as st
#import seaborn as sns
import os 

start = '2010-01-01'
end='2022-10-22'



page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{{
   background-image: url("https://www.istockphoto.com/photo/world-map-and-networking-gm1346580665-424314091?utm_source=unsplash&utm_medium=affiliate&utm_campaign=category_photos_top&utm_content=https%3A%2F%2Funsplash.com%2Fbackgrounds&utm_term=Hq%20background%20images%3A%3A%3A");
   background-size:cover;
}}
</style>
"""
with open("style.css") as source_des:
    st.markdown(f"<style>{source_des.read()}</style>",unsafe_allow_html=True)

import streamlit.components.v1 as components  # Import Streamlit

# Render the h1 block, contained in a frame of size 200x200.
#components.html('<html><body><h1 style="color:red;">Hello, Welcome to our Stock Market Price Prediction App</h1></body></html>') 
components.html("""<html>
<style>
h1.head{
    color:red;
    background-color:white;
}
</style>
<body>
<h1 class="head">Hello, Welcome to our Stock Market Price Prediction App</h1>
</body>
</html>""")   

st.title('Stock Trend Prediction')
user_input=st.text_input('Enter Stock Ticker','AAPL')
df=data.DataReader(user_input,'yahoo',start,end)


#Describing Data
st.subheader('Data from 01-01-2010 to 22-10-2022')
st.write(df.describe())


#timly graph for any stock.
st.subheader(f'Last year Graph for {user_input} Stock')
fig=plt.figure(figsize=(20,10))
df['Close'].plot(xlim=['2021-01-01','2022-10-10'],ylim=[100,200],figsize=(20,10),c='red')
df['Open'].plot(xlim=['2021-01-01','2022-10-10'],ylim=[100,200],figsize=(20,10),c='green')
# plt.plot(df.Close)
plt.xlabel("Dates", fontsize=14)
plt.ylabel("prices", fontsize=14)
plt.legend()
plt.grid()
st.pyplot(fig)



st.subheader(f'Past two year opening price vs index/date Graph for {user_input} Stock')
index = df.loc['2020-01-01':'2022-10-28'].index
share_open=df.loc['2020-01-01':'2022-10-28']['Open']
share_close=df.loc['2020-01-01':'2022-10-28']['Close']

fig=plt.figure(figsize=(20,10))
figure,axis=plt.subplots()
# plt.tight_layout()
figure.autofmt_xdate()
plt.xlabel("Dates", fontsize=14)
plt.ylabel("opening prices", fontsize=14)
plt.legend()
axis.plot(index,share_open)
plt.grid()
plt.savefig('x',dpi=400)
st.image('x.png')
os.remove('x.png')


st.subheader(f'Past two year closing price vs index/date Graph for {user_input} Stock')
index = df.loc['2020-01-01':'2022-10-28'].index
share_open=df.loc['2020-01-01':'2022-10-28']['Open']
share_close=df.loc['2020-01-01':'2022-10-28']['Close']

fig=plt.figure(figsize=(12,10))
figure,axis=plt.subplots()
# plt.tight_layout()
plt.xlabel("Dates", fontsize=14)
plt.ylabel("closing prices", fontsize=14)
plt.legend()
figure.autofmt_xdate()
axis.plot(index,share_close)
plt.grid()
plt.savefig('x',dpi=400)
st.image('x.png')
os.remove('x.png')


from datetime import datetime

date = datetime(2021,11,21)

miniz = df.resample(rule='A').min()



st.subheader(f'Minimum price yearly Graph for {user_input} Stock')
fig=plt.figure(figsize=(20,10))
plt.plot(miniz.Close)
plt.xlabel("Dates", fontsize=14)
plt.ylabel("Minimum prices", fontsize=14)
plt.legend()
plt.grid()
st.pyplot(fig)


maxiz = df.resample(rule='A').max()

st.subheader(f'Maximum price yearly Graph for {user_input} Stock')
fig=plt.figure(figsize=(12,10))
plt.plot(maxiz.Close)
plt.xlabel("Dates", fontsize=14)
plt.ylabel("Maximum prices", fontsize=14)
plt.legend()
plt.grid()
st.pyplot(fig)

st.subheader(f'Maximum price montly Graph for {user_input} Stock')
fig=plt.figure(figsize=(12,10))
montly_maxiz = df.resample(rule='M').max()
plt.xlabel("Dates", fontsize=14)
plt.ylabel("Maximum prices", fontsize=14)
plt.legend()
plt.plot(montly_maxiz.Close)
plt.grid()
st.pyplot(fig)


st.subheader(f'Maximum price Business yearly Graph for {user_input} Stock')
fig=plt.figure(figsize=(12,10))
df.resample(rule='BA').max()['Close'].plot()
plt.xlabel("Dates", fontsize=14)
plt.ylabel("Maximum prices", fontsize=14)
plt.legend()
plt.grid()
st.pyplot(fig)

st.subheader(f'Maximum price Business yearly Bar Graph for {user_input} Stock')
fig=plt.figure(figsize=(12,10))
df.resample(rule='BA')['Close'].mean().plot(kind='bar')
plt.xlabel("Dates", fontsize=14)
plt.ylabel("Maximum prices", fontsize=14)
plt.grid()
st.pyplot(fig)

st.subheader(f'Business End {user_input}')
fig=plt.figure(figsize=(12,10))
df.resample(rule='BQS').max()['Close'].plot()
plt.xlabel("Dates", fontsize=14)
plt.ylabel("Maximum prices", fontsize=14)
plt.grid()
st.pyplot(fig)



st.subheader(f'Business Quarter {user_input}')
fig=plt.figure(figsize=(12,10))
df.resample(rule='BQS')['High'].mean().plot(kind='bar',figsize=(15,6))
plt.xlabel("Dates", fontsize=14)
plt.ylabel("Prices", fontsize=14)
plt.grid()
st.pyplot(fig)


import seaborn as sns


st.subheader('Heatmap Between Attributes')
fig=plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),cmap='gray',annot=True)
st.pyplot(fig)
st.text('From the above heatmap, we can see a large number \nof 1s and values close to 1. This means those variables show high \npositive correlations and are interrelated. \nThis might be possible because of the comparatively very small difference\nbetween those values. However,in-stock market this small value is \nwhat makes the difference')



st.subheader('Open and Volumn Attribute Relationship')
fig=plt.figure(figsize=(12,10))
sns.barplot(data=df, x = "Open", y="Volume")
st.text('From the below graph, we can observe that the volume is high \n for smaller values of open-high as compared to larger values of\n open-high.')

st.pyplot(fig)


mean = df['Close'].mean()
std = df['Close'].std()


st.subheader(f'Analyzing mean and std with the help of graph for {user_input}')
fig=plt.figure(figsize=(12,10))
df['Close'].hist(bins=20)
plt.axvline(mean,color='red',linestyle='dashed',linewidth=2)
plt.axvline(std,color='g',linestyle='dashed',linewidth=2)
plt.axvline(-std,color='g',linestyle='dashed',linewidth=2)
plt.savefig('x',dpi=400)
st.image('x.png')
os.remove('x.png')
st.text('to plot the std line we plot both the positive and negative values')


#Visualizations
st.subheader('Closing Price vs Time chart')
fig=plt.figure(figsize=(12,10))
plt.plot(df.Close,label='closing price') 
plt.xlabel("Dates", fontsize=14)
plt.ylabel("closing prices", fontsize=14)
plt.legend()
plt.grid()
st.pyplot(fig)

st.subheader('Closing Price vs Time chart 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,10)) 
st.text('closing moving average of only 100 days')
plt.plot(ma100,label='MA 100')
plt.xlabel("Dates", fontsize=14)
plt.ylabel("Price", fontsize=14)
plt.legend()
plt.plot(df.Close,label='closing_price')
plt.grid()
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,10))
plt.plot(ma100,'g',label='MA 100')
plt.plot(ma200,'r',label='MA 200')
plt.xlabel("Dates", fontsize=14)
plt.ylabel("Prices", fontsize=14)
plt.legend()
plt.plot(df.Close,'b',label='closing_price')
plt.grid()
st.pyplot(fig)

req_col=pd.DataFrame(df['Close'])
col=np.array(req_col)

train_size = int(len(req_col)*0.7)
test_size = (len(col)-train_size)
train_data,test_data = col[0:train_size,:],col[train_size:len(col),:1]

scaler=MinMaxScaler(feature_range=(0,1))
data_training_array=scaler.fit_transform(train_data)

#load my model

model = load_model('keras_model1.h5')


from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential

past_100_days = train_data[-100:]

import numpy

final_df=numpy.concatenate((past_100_days,test_data))
input_data=scaler.fit_transform(final_df)



input_test=[]
output_test=[]

for i in range(100,len(input_data)):
    input_test.append(input_data[i-100:i])
    output_test.append(input_data[i,0])

input_test, output_test = np.array(input_test), np.array(output_test)

output_predicted=model.predict(input_test)

scaler = scaler.scale_

scale_factor=1/scaler[0]

output_predicted=output_predicted*scale_factor
output_test=output_test*scale_factor

#final graph
components.html("""<html>
<style>
h1.head{
    color:red;
    background-color:white;
}
</style>
<body>
<h1 class="head">The below gragh is final graph between original stock price vs predicted price</h1>
</body>
</html>""")

st.subheader('Predication vs Original')
fig2 = plt.figure(figsize=(12,10))
plt.plot(output_test,'b',label='Original Price')
plt.plot(output_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid()
st.pyplot(fig2)



components.html("""<html>
<style>
h1.head{
    color:red;
    background-color:white;
}
</style>
<body>
<h1 class="head">Output in 50 epochs training above one seems overfitting.</h1>
</body>
</html>""")

from PIL import Image

image=Image.open('z.png')
st.image(image,caption='output in 50 epochs training above one seems overfitting.')




df1=data.DataReader('META','yahoo',start,end)
df2=data.DataReader('AMZN','yahoo',start,end)
df3=data.DataReader('AAPL','yahoo',start,end)
df4=data.DataReader('NFLX','yahoo',start,end)
df5=data.DataReader('GOOGL','yahoo',start,end)

close_price=pd.DataFrame()

close_price['facebook']=df1.Close
close_price['amazon']=df2.Close
close_price['apple']=df3.Close
close_price['netflix']=df4.Close
close_price['google']=df5.Close


components.html("""<html>
<style>
h1.head{
    color:red;
    background-color:white;
}
</style>
<body>
<h1 class="head">Below we are comparing FAANG companies' stock.</h1>
</body>
</html>""")
#st.subheader("Comparing FAANG commany's stock.")
fig1=plt.figure(figsize=(12,10))
sns.pairplot(data=close_price,height=3.5)
plt.savefig('x',dpi=400)
st.image('x.png')
os.remove('x.png')


st.subheader('Closing Price Co-Relation')
fig=plt.figure(figsize=(12,10))
sns.heatmap(close_price.corr(),annot=True,cmap='gray_r',linecolor="black")
st.pyplot(fig)
st.text(' result - Closing price of apple and google have a co-relation of 0.96.')
