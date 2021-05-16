#!/usr/bin/env python
# coding: utf-8

# ## とりあえず表示させる

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import mplfinance as mpf
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('usd_10min_api.csv', index_col=0, parse_dates=True)
print(df)
df['time'] = pd.to_datetime(df['time'])
print(df['time'].dtype)
df = df.set_index('time')
df = df.reindex(columns=['open', 'high', 'low', 'close', 'volume'])
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
df.head(3)


# In[3]:


mpf.plot(df[:100], type='candle', figratio=(12,4))


# ## 酒田五法の実装

# In[20]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import mplfinance as mpf
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


df = pd.read_csv('usd_10min_api.csv', index_col=0, parse_dates=True)
print(df)
df['time'] = pd.to_datetime(df['time'])
print(df['time'].dtype)
df = df.set_index('time')
print(df)
df = df.reindex(columns=['open', 'high', 'low', 'close', 'volume'])
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
print(df)
df2 = df.copy()


# In[22]:


mpf.plot(df[1000:1250], type='candle', figratio=(18,9),volume=True, mav=(5, 25), style='yahoo')


# In[23]:


df2['Close10'] = df2['Close'].shift(1)
df2['Close20'] = df2['Close'].shift(2)
df2['Close30'] = df2['Close'].shift(3)
df2['Open10'] = df2['Open'].shift(1)
df2['Open20'] = df2['Open'].shift(2)
df2['Open30'] = df2['Open'].shift(3)
df2


# # ポリンジャーバンドの計算

# In[24]:


from pyti.bollinger_bands import upper_bollinger_band as bb_up
from pyti.bollinger_bands import middle_bollinger_band as bb_mid
from pyti.bollinger_bands import lower_bollinger_band as bb_low

data = df2['Close'].values.tolist()
period = 20
bb_up = bb_up(data,period)
bb_mid = bb_mid(data,period)
bb_low = bb_low(data,period)
df2['bb_up'] = bb_up
df2['bb_mid'] = bb_mid
df2['bb_low'] = bb_low

df3 = df2[1000:1250]
df3[['Close','bb_up','bb_mid','bb_low']].plot()


# In[25]:


apd = mpf.make_addplot(df2[550:750][['bb_up', 'bb_mid', 'bb_low']])
mpf.plot(df2[550:750], type='candle', addplot=apd, volume=True,style='yahoo')


# ## 赤三兵を検出する

# In[52]:


# ルールその1 Close30 < Low
df2['rule_1'] = 0
rule_1_mask = df2['Close'] < ((df2['bb_low']+df2['bb_mid'])/2)
df2['rule_1'][rule_1_mask] = 1
df2.head()


# In[53]:


# ルールその2
df2['rule_2'] = 0
rule_2_mask = (df2['Open'] - df2['Close'] < 0) & (df2['Open10'] - df2['Close10'] < 0) & (df2['Open20'] - df2['Close20'] < 0)
df2['rule_2'][rule_2_mask] = 1


# In[54]:


# ルール1とルール2が該当するレコードを探す
df2[(df2['rule_1'] == 1.0) & (df2['rule_2'] == 1.0)]


# In[55]:


# 酒田五法 赤三兵の確認 その1
candle_temp = df2[1810:1890]
apd = mpf.make_addplot(candle_temp[['bb_up', 'bb_mid', 'bb_low']])
mpf.plot(candle_temp, type='candle', addplot=apd, volume=True,style='yahoo')


# # ada/jpyで表示

# In[56]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import mplfinance as mpf
get_ipython().run_line_magic('matplotlib', 'inline')


# In[65]:


df = pd.read_csv('ada_jpy_charts.csv', index_col=0, parse_dates=True)
#print(df)
df = df.reindex(columns=['open', 'high', 'low', 'close', 'volume','weighted'])
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume','Weighted']
df.head(3)


# In[58]:


#savefig = 'ada_jpy_chart_yahoo'を引数に追加でファイル出力
#mpf.plot(df, type='candle', figratio=(18,9),volume=True, mav=(5, 25), style='yahoo')


# In[66]:


df['Close1'] = df['Close'].shift(1)
df['Close2'] = df['Close'].shift(2)
df['Close3'] = df['Close'].shift(3)
df['Open1'] = df['Open'].shift(1)
df['Open2'] = df['Open'].shift(2)
df['Open3'] = df['Open'].shift(3)


# In[67]:


from pyti.bollinger_bands import upper_bollinger_band as bb_up
from pyti.bollinger_bands import middle_bollinger_band as bb_mid
from pyti.bollinger_bands import lower_bollinger_band as bb_low

data = df['Close'].values.tolist()
period = 20
bb_up = bb_up(data,period)
bb_mid = bb_mid(data,period)
bb_low = bb_low(data,period)
df['bb_up'] = bb_up
df['bb_mid'] = bb_mid
df['bb_low'] = bb_low


# In[68]:


apd = mpf.make_addplot(df[['bb_up', 'bb_mid', 'bb_low']])
mpf.plot(df, type='candle', addplot=apd, volume=True,style='yahoo')


# In[69]:


# ルールその1 Close3 < Low
df['rule_1'] = 0
rule_1_mask = df['Close3'] < ((df['bb_low']+df['bb_mid'])/2)
df['rule_1'][rule_1_mask] = 1


# In[70]:


# ルールその2
df['rule_2'] = 0
rule_2_mask = (df['Open'] - df['Close'] < 0) & (df['Open1'] - df['Close1'] < 0) & (df['Open2'] - df['Close2'] < 0)
df['rule_2'][rule_2_mask] = 1


# In[71]:


# ルール1とルール2が該当するレコードを探す
df[(df['rule_1'] == 1.0) & (df['rule_2'] == 1.0)]


# In[72]:


# 酒田五法 赤三兵の確認 その1
start_date ='2020-09-28 09:09:00'
end_date = '2020-11-22 09:11:00'
candle_temp = df.loc[start_date:end_date]
apd = mpf.make_addplot(candle_temp[['bb_up', 'bb_mid', 'bb_low']])
mpf.plot(candle_temp, type='candle', addplot=apd, volume=True,style='yahoo')


# # 三空叩き込み
# - 下落相場
# - 陰線が四本連続
# - マドが空いている

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import mplfinance as mpf
get_ipython().run_line_magic('matplotlib', 'inline')

from pyti.bollinger_bands import upper_bollinger_band as bb_up
from pyti.bollinger_bands import middle_bollinger_band as bb_mid
from pyti.bollinger_bands import lower_bollinger_band as bb_low

df = pd.read_csv('ada_jpy_charts.csv', index_col=0, parse_dates=True)
df = df.reindex(columns=['open', 'high', 'low', 'close', 'volume','weighted'])
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume','Weighted']

df['Close1'] = df['Close'].shift(1)
df['Close2'] = df['Close'].shift(2)
df['Close3'] = df['Close'].shift(3)
df['Open1'] = df['Open'].shift(1)
df['Open2'] = df['Open'].shift(2)
df['Open3'] = df['Open'].shift(3)

data = df['Close'].values.tolist()
period = 20
bb_up = bb_up(data,period)
bb_mid = bb_mid(data,period)
bb_low = bb_low(data,period)
df['bb_up'] = bb_up
df['bb_mid'] = bb_mid
df['bb_low'] = bb_low


# # 決定木で予想する

# In[20]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#import graphviz


# In[21]:


df = pd.read_csv('usd_jpy_api.csv', index_col=0, parse_dates=True)
df.tail(5)


# In[9]:


df_5_mean = df.rolling(window=5).mean()
df_5_mean.head(5)


# In[12]:


df[0:5].sum() / 5


# In[3]:


df['close+1'] = df.close.shift(-1)
df['diff'] = df['close+1'] - df['close']
df = df[:-1]
df.tail()


# In[4]:


m = len(df['close'])
print(len(df[(df['diff'] > 0)]) / m*100)
print(len(df[(df['diff'] <0)]) /m*100)


# In[5]:


mask1 = df['diff'] > 0
mask2 = df['diff'] < 0
column_name = 'diff'
df.loc[mask1, column_name] = 1
df.loc[mask2,column_name] = 0
df.head()


# In[6]:


df.rename(columns={'diff':'target'},inplace=True)

del df['close+1']
df = df[['target', 'close', 'open', 'high', 'low', 'volume']]
df.head()


# In[15]:


# データセットの行数、列数を取得
n = df.shape[0]
p = df.shape[1]
print(n)
# 訓練データとテストデータへ分割
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end + 1
test_end = n
# ilocは数字で指定、locは列名で指定、at,iatは単独指定だが高速
data_train = df.iloc[train_start : train_end, :]
data_test = df.iloc[test_start : test_end, :]
 
# 訓練データとテストデータのサイズを確認
print(data_train.shape)
print(data_test.shape)


# In[ ]:




