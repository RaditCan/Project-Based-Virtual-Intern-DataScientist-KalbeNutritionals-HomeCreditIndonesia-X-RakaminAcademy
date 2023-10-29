#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df_transaction = pd.read_csv('Case Study - Transaction.csv', delimiter = ';')
df_transaction.head()


# In[3]:


df_customer = pd.read_csv('Case Study - Customer.csv',delimiter = ';')
df_customer.head()


# In[4]:


df_product = pd.read_csv('Case Study - Product.csv',delimiter = ';')
df_product.head()


# In[5]:


df_store = pd.read_csv('Case Study - Store.csv',delimiter = ';')
df_store.head()


# ### Data Cleansing

# In[6]:


#Customer
df_customer['Income'] = df_customer['Income'].replace('[,]','.',regex=True).astype('float')
df_customer


# In[7]:


# Store
df_store['Latitude'] = df_store['Latitude'].replace('[,]','.',regex=True).astype('float')
df_store['Longitude'] = df_store['Longitude'].replace('[,]','.',regex=True).astype('float')
df_store


# In[8]:


# transaction
df_transaction['Date']= pd.to_datetime(df_transaction['Date'])
df_transaction


# ### Merger Data

# In[9]:


df_merge = pd.merge(df_transaction,df_customer, on = ['CustomerID'])
df_merge = pd.merge(df_merge,df_product.drop(columns=['Price']),on=['ProductID'])
df_merge = pd.merge(df_merge,df_store,on=['StoreID'])
df_merge.head()


# ### Machine Learning Regression (Time Series)

# In[10]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot

import statsmodels.api as sm
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from kneed import DataGenerator, KneeLocator
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import warnings
warnings.filterwarnings('ignore')


# In[11]:


df_regresion = df_merge.groupby('Date').agg({'Qty':'sum'}).reset_index()
df_regresion


# In[12]:


data_index = df_regresion.set_index('Date')
regresion_decomposition = seasonal_decompose(data_index)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

regresion_decomposition.trend.plot(ax=ax1)
ax1.set_ylabel('Trend', fontsize=18)
regresion_decomposition.seasonal.plot(ax=ax2)
ax2.set_ylabel('Seasonal', fontsize=18)
regresion_decomposition.resid.plot(ax=ax3)
ax3.set_ylabel('Residual', fontsize=18)

plt.tight_layout()
plt.show()


# ### Uji Stasioneritas

# In[13]:


from statsmodels.tsa.stattools import adfuller

result = adfuller(df_regresion['Qty'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
  print('\t%s: %.3f' % (key, value))

print("% significant : 0.05")
print("Conclusion :")
if result[1] <= 0.05:
  print("The data is stationary (",result[1],"<= 0.05, reject H0)")
else:
  print("The data is non-stationary (",result[1],"> 0.05, fail to reject H0)")


# #### Karena stasioner, maka kemungkinan besar model ARIMA(p,0,q)

# #### Nilai p dan q

# In[14]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

plot_acf(df_regresion.Qty.dropna())


# In[15]:


sm.graphics.tsa.plot_pacf(df_regresion.Qty.dropna())


# ##### dari PACF dan ACF disimpulkan bahwa model adalah ARIMA(0,0,0)

# ##### dari PACF juga tidak terdapat pola tertentu yang berarti tidak ada tanda2 seasonal

# In[16]:


# split 80% data training dan 20% data testing
cut_off = round(df_regresion.shape[0] * 0.8) 
df_train = df_regresion[:cut_off]
df_test = df_regresion[cut_off:].reset_index(drop=True)
df_train.shape, df_test.shape


# In[17]:


df_train


# In[18]:


df_test


# In[19]:


plt.figure(figsize=(20,5))
sns.lineplot(data=df_train, x=df_train['Date'], y=df_train['Qty'])
sns.lineplot(data=df_test, x=df_test['Date'], y=df_test['Qty'])


# In[20]:


#RMSE dan MAE
def rmse (y_actual, y_pred):
  print(f'RMSE Value{mean_squared_error(y_actual, y_pred, squared=False)** 0.5}')

def eval(y_actual, y_pred):
  rmse(y_actual, y_pred)
  print(f'MAE Value{mean_absolute_error(y_actual, y_pred)** 0.5}')


# In[21]:


#ARIMA
df_train = df_train.set_index('Date')
df_test = df_test.set_index('Date')
y = df_train['Qty']

model_ARIMA = ARIMA(y, order=(0,0,0)) #order = (p,d,q)
model_ARIMA = model_ARIMA.fit()

y_pred = model_ARIMA.get_forecast(len(df_test))

y_pred_df = y_pred.conf_int()
y_pred_df['prediction'] = model_ARIMA.predict(start = y_pred_df.index[0],end = y_pred_df.index[-1])
y_pred_df.index = df_test.index
y_pred_out = y_pred_df['prediction']

eval(df_test['Qty'],y_pred_out)

plt.figure(figsize=(10,8))
plt.plot(df_train['Qty'])
plt.plot(df_test['Qty'],color = 'red')
plt.plot(y_pred_out, color = 'black', label = 'ARIMA Prediction')
plt.legend()
plt.tight_layout()


# In[22]:


# Melakukan Forecast dari data yang digunakan selama satu bulan = 31 hari
forecast_result = model_ARIMA.get_forecast(31)
forecast_result = forecast_result.conf_int()
forecast_result['forecasted Qty'] = model_ARIMA.predict(start = forecast_result.index[0], end = forecast_result.index[-1])
forecast_result['Date'] = pd.date_range(start ='2023-01-01', end = '2023-01-31')
forecast_result.set_index('Date', inplace = True)
forecast_result.head()


# In[23]:


plt.figure(figsize = (10,8))
plt.plot(df_train['Qty'])
plt.plot(df_test['Qty'], color = 'red')
plt.plot(y_pred_out, color = 'black', label = 'ARIMA Predictions')
plt.plot(forecast_result['forecasted Qty'], color = 'yellow', label = 'ARIMA Forecasted')
plt.legend()
plt.tight_layout()


# ### Clustering

# In[24]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[25]:


df_merge


# In[26]:


df_merge.corr()


# In[27]:


df_cluster = df_merge.groupby('CustomerID').agg({'TransactionID':'count', 'Qty':'sum','TotalAmount':'sum'}).reset_index()
df_cluster


# In[28]:


plt.figure(figsize=(15,3))

plt.subplot(1,2,1)
sns.histplot(df_cluster['Qty'],color='royalblue',kde=True)
plt.title('distribusi Qty', fontsize=16)
plt.xlabel('Qty', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

plt.subplot(1,2,2)
sns.histplot(df_cluster['TotalAmount'],color='deeppink', kde=True)
plt.title('distribusi TotalAmount', fontsize=16)
plt.xlabel('TotalAmount', fontsize=14)

plt.show()


# In[29]:


plt.figure(figsize=(15,3))

plt.subplot(1,2,1)
sns.histplot(df_cluster['TransactionID'],color='deeppink', kde=True)
plt.title('distribusi TransactionID', fontsize=16)
plt.xlabel('TransactionID', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

plt.show()


# In[30]:


data_cluster = df_cluster.drop(columns=['CustomerID'])
data_cluster_normal = MinMaxScaler().fit_transform(data_cluster)
data_cluster_normal


# In[31]:


df_cluster_normal = pd.DataFrame(data = data_cluster_normal, columns=data_cluster.columns)
df_cluster_normal


# #### Menentukan jumlah K

# In[32]:


WCSS=[]
n = range(2 , 11)
for i in n:
    kmeans=KMeans(n_clusters=i, random_state=0)
    kmeans.fit(data_cluster_normal)
    WCSS.append(kmeans.inertia_)
print(WCSS)


# In[33]:


plt.figure(figsize=(8,3))
plt.plot(list(n), WCSS, color='royalblue', marker='o',linewidth=2,markersize=12,markerfacecolor='m', markeredgecolor='m')
plt.title('WCSS VS Banyaknya Cluster', fontsize=18)
plt.xlabel('Jumlah Cluster', fontsize=15)
plt.ylabel('WCSS', fontsize=15)
plt.show()


# #### sudah tidak terjadi perubahan yang signifikan ketika pada k=3, karena hanya selisih 2 dari k=4

# In[34]:


# Clustering dengan k = 3
kmeans = KMeans(n_clusters=3, random_state=0)
df_cluster['Cluster']= kmeans.fit_predict(df_cluster)
df_kmeans = df_cluster
df_kmeans


# In[35]:


#rata-rata per cluster
mean_cluster = df_kmeans.groupby(('Cluster')).agg({
    'CustomerID':'count',
    'TransactionID':'mean',
    'Qty':'mean',
    'TotalAmount':'mean'
})
mean_cluster


# In[36]:


#jumlah cluster
cluster_count = df_cluster['Cluster'].value_counts()
print('Jumlah cluster :')
print(cluster_count)


# In[37]:


#Visualisasi
plt.figure(figsize=(18,10))
sns.set_style('white')
plt.scatter(x=df_kmeans['TransactionID'],y=df_kmeans['Qty'],c=df_kmeans['Cluster'],cmap='Paired')
plt.title('Kmeans Clustering', fontsize=24)
plt.xlabel('TransactionID', fontsize=18)
plt.ylabel('Qty', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid()
plt.show()


# In[38]:


print(f'Silhouette Score(n=3):',{silhouette_score(df_cluster_normal, df_kmeans['Cluster'])})


# In[39]:


mean_cluster


# ### Interpretasi Clustering

# Interpretasi 3 kelompok cluster adalah berikut ini:
# 
# 1. Cluster 0 - pada cluster 0 terdapat total Customer sebanyak 90 dengan rata-rata transaksi sebanyak 15 kali, sedangkan total kuantitas produk yang di beli yaitu rata-rata 58 produk dan rata-rata jumlah uang yang di belanjakan Customer adalah sekitar Rp548.162,- yang berarti total uang yang dibelanjakan 90 Customer adalah 49 juta atau Rp49.334.580,-
# 
# 2. Cluster 1 - pada cluster 1 terdapat total Customer sebanyak 186 dengan rata-rata transaksi sebanyak 11 kali, sedangkan total kuantitas produk yang di beli yaitu rata-rata 42 produk dan rata-rata jumlah uang yang di belanjakan Customer adalah sekitar Rp384.003,- yang berarti total uang yang dibelanjakan 186 Customer adalah 71 juta atau Rp71.424.558,-
# 
# 3. Cluster 2 - pada cluster 2 terdapat total Customer sebanyak 171 dengan rata-rata transaksi sebanyak 8 kali, sedangkan total kuantitas produk yang di beli yaitu rata-rata 29 produk dan rata-rata jumlah uang yang di belanjakan Customer adalah sekitar Rp241.425,- yang berarti total uang yang dibelanjakan 171 Customer adalah 41 juta atau Rp41.283.675,-
# 
# ### Rekomendasi Bisnis
# 
# 1. Cluster 2 - pada cluster 2 jumlah kuantitas produk yang paling sedikit dari cluster lainnya, meskipun jumlah Customer cukup banyak hal ini berarti terdapat kurangnya daya beli Customer cluster 2 sehingga perusahaan dapat memberikan diskon dan promo pada tingkat tertentu untuk menarik Customer membeli produk. Selain itu, perusahaan (Marketing) juga harus melakukan branding lagi untuk mendorong penjualan.
# 
# 2. Cluster 1 - pada cluster 1 jumlah Customer dan total uang yang dibelanjakan paling banyak dari cluster lainnya. Maka prioritas utama perusahaan pada cluster 1 adalah mempertahankan Customer langganan untuk tetap setia pada perusahaan dan menjaga kepercayaan Customer dengan menjaga sumber, kualitas, dan keamanan produk. Selain itu, bagian marketing dapat menawarkan promo dan diskon melalui Email sehingga perusahaan dan customer dapat menjaga relasi.
# 
# 3. Cluster 0 - pada cluster 0 jumlah rata-rata produk dan rata-rata uang yang dibelanjakan per Customer adalah yang terbanyak dari cluster lainnya, meskipun jumlah Customernya paling sedikit. Hal ini dapat berarti Customer memiliki daya beli yang tinggi. Daya beli yang tinggi diikuti dengan konsumsi yang tinggi. Oleh karena itu, perusahaan dapat memberikan kartu anggota bagi yang mencapai target belanja tertentu dan menawarkan produk exclusive bagi anggota.
# 

# In[ ]:




