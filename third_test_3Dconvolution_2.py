from tensorflow.keras import Sequential
#from tensorflow.keras import Dense, Flatten, Conv3D, MaxPooling3D, Dropout
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
# -- Preparatory code --
#相關參數設置
shape = 15, 29, 35, 3 #35,29,15
batch_size = 40
no_epochs = 50
learning_rate = 0.001
validation_split = 0.2
verbosity = 1


# Convert 1D vector into 3D values, provided by the 3D MNIST authors at
# https://www.kaggle.com/daavoo/3d-mnist
def array_to_color(array, cmap="Oranges"):
    s_m = plt.cm.ScalarMappable(cmap=cmap)
    return s_m.to_rgba(array)[:,:-1]


# Reshape data into format that can be handled by Conv3D layers.
# Courtesy of Sam Berglin; Zheming Lian; Jiahui Jang - University of Wisconsin-Madison
# Report - https://github.com/sberglin/Projects-and-Papers/blob/master/3D%20CNN/Report.pdf
# Code - https://github.com/sberglin/Projects-and-Papers/blob/master/3D%20CNN/network_final_version.ipynb
def rgb_data_transform(data):
    data_t = []
    for i in range(data.shape[0]):
        data_t.append(array_to_color(data[i]).reshape(shape))
    return np.asarray(data_t, dtype=np.float32)

#%%

#載入csv
data = pd.read_csv("D:/00Dissertation_Discuss/file/third_test/testttt_merge_fixed.csv")
data_copy = data.copy()
data_clean = data_copy.drop(columns = data_copy.columns[[0]])

#%%

#劃分輸入與輸出
data_output = data_clean.iloc[:, -3:]
data_input = data_clean.iloc[:, :-3]



#劃分訓練集與測試集
X_train = data_input.sample(frac=0.8,random_state=0)
X_test = data_input.drop(X_train.index)

#取出預測值
targets_train = data_output.iloc[X_train.index]
targets_test = data_output.drop(X_train.index)



#input 標準化
#查看資料整體狀況
train_stats = X_train.describe()
train_stats = train_stats.transpose()

#def norm(x):
#  return (x - train_stats['mean']) / train_stats['std']
#normed_X_train = norm(X_train).fillna(value=0)
#normed_X_test = norm(X_test).fillna(value=0)



#output 標準化

train_label_stats = targets_train.describe()
train_label_stats = train_label_stats.transpose()

def norm(x):
  return (x - train_label_stats['min']) / (train_label_stats['max']-train_label_stats['min'])
normed_targets_train = norm(targets_train).fillna(value=0)
normed_targets_test = norm(targets_test).fillna(value=0)



#轉換資料格式
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
targets_train = normed_targets_train.to_numpy()
targets_test = normed_targets_test.to_numpy()

pd_X_train = pd.DataFrame(X_train)
pd_targets_train = pd.DataFrame(targets_train)
pd_X_test = pd.DataFrame(X_test)
pd_targets_test = pd.DataFrame(targets_test)


#賦予資料顏色通道
X_train = rgb_data_transform(X_train)
X_test = rgb_data_transform(X_test)


#%%
#卷積模型建構
sample_shape = tuple(shape)
kernel_size = (3, 3, 3)
kernel_size2 = (3, 3, 3)

model = Sequential([
    layers.Conv3D(64, kernel_size=kernel_size, activation='relu', input_shape=sample_shape, padding='same'),
#    layers.Conv3D(16, kernel_size=kernel_size, activation='relu', padding='same'),
    layers.MaxPooling3D(),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    
    layers.Conv3D(64, kernel_size=kernel_size2, activation='relu', padding='same'),
#    layers.Conv3D(64, kernel_size=kernel_size2, activation='relu', padding='same'),
    layers.MaxPooling3D(),
    layers.BatchNormalization(),
    layers.Dropout(0.4),    
    

#    layers.Conv3D(32, kernel_size=kernel_size2, activation='relu', padding='same'),
    layers.Conv3D(128, kernel_size=kernel_size2, activation='relu', padding='same'),
    layers.MaxPooling3D(),
    layers.BatchNormalization(),
    layers.Dropout(0.4),        

    
    layers.Flatten(),
    layers.Dense(5000, activation='relu'),
    layers.Dropout(0.4),

    
    layers.Dense(5000, activation='relu'),
    layers.Dropout(0.4),
    
    layers.Dense(3)
    ])

optimizer = 'adam'
# Compile the model
model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])

model.summary()

#%%
#訓練模型
history = model.fit(X_train, targets_train,
                    batch_size=batch_size,
                    epochs=no_epochs,
                    verbose=verbosity,
                    validation_split=validation_split)


#%%
#使用 history 对象中存储的统计信息可视化模型的训练进度
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

#繪製學習曲線
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error ')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([hist["mae"].min(),hist["val_mae"].max()/6])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error ')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([hist["mse"].min(),hist["val_mse"].max()/10])
  plt.legend()
  plt.show()


plot_history(history)

#%%
test_predictions = model.predict(X_test)


#繪製
x_ax = range(len(X_test))
normed_test_predictions = pd.DataFrame(test_predictions)


#繪製比較關係
plt.scatter(pd.DataFrame(targets_test).iloc[:,0], normed_test_predictions.iloc[:,0], color='r')
plt.xlabel('True Values [Sun_average]')
plt.ylabel('Predictions [Sun_average]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,1.5])
plt.ylim([0,1.5])
_= plt.plot([-100, 100], [-100, 100], 'r')
plt.show()


plt.scatter(pd.DataFrame(targets_test).iloc[:,1], normed_test_predictions.iloc[:,1], color='b')
plt.xlabel('True Values [total_radiation]')
plt.ylabel('Predictions [total_radiation]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,1.5])
plt.ylim([0,1.5])
_= plt.plot([-100, 100], [-100, 100], 'b')
plt.show()


plt.scatter(pd.DataFrame(targets_test).iloc[:,2], normed_test_predictions.iloc[:,2], color='g')
plt.xlabel('True Values [visibility]')
plt.ylabel('Predictions [visibility]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,1.5])
plt.ylim([0,1.5])
_ = plt.plot([-100, 100], [-100, 100], 'g')
plt.show()

#%%
#保存模型
model.save('./keras_model.h5')



