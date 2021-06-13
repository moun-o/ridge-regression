

```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


data=pd.read_csv('dataset.csv')

#replace inf by nan
data=data.replace([np.inf, -np.inf], np.nan)


```


```python
#interpolate nan data
data=data.interpolate(method ='linear', limit_direction ='backward')
data.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>-10.149463</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.025183</td>
      <td>-10.149463</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.050366</td>
      <td>-7.517911</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.075549</td>
      <td>-5.480920</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.100732</td>
      <td>-5.882184</td>
    </tr>
  </tbody>
</table>
</div>




```python
x=np.array(data['x']).reshape(np.size(data['x']),1)
y=np.array(data['y'])
```


```python
plt.figure()
plt.scatter(x[:, 0], y,alpha=0.3,label='data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()  
```


![png](output_3_0.png)



```python
"""
In this case we notice that our data is not linear, so we can't pply linear regression, the best method for this sinusoidal
shape looking data is kernelRidge regression with an rbf Kernel

"""
```




    "\nIn this case we notice that our data is not linear, so we can't pply linear regression, the best method for this sinusoidal\nshape looking data is kernelRidge regression with an rbf Kernel\n\n"




```python
from sklearn.kernel_ridge import KernelRidge
```


```python
reg = KernelRidge(kernel='rbf')
reg.fit(x, y)
p=reg.predict(x)
```


```python
plt.figure()
plt.title(reg.__class__.__name__)
plt.scatter(x[:, 0], y,alpha=0.3,label='data')
plt.plot(x, p,color='RED',label='Kernel Ridge Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})
plt.show()   
"""
Here we can see the approximation of the regression function
"""
```


![png](output_7_0.png)





    '\nHere we can see the approximation of the regression function\n'




```python
"""
In order to answer the given problematic to 15, we suggest to generate the x values to 15, using a mean step
"""
```




    '\nIn order to answer the given problematic to 15, we suggest to generate the x values to 15, using a mean step\n'




```python
#compute step to generate x : the mean of  the difference between each element of x with his successor.
x_step=np.mean(np.diff(x.ravel()))

#init min & max as requested
x_max=15
x_min=np.max(x)

#extend x
x_extend=np.arange(start=np.max(x), stop=15, step=x_step)
x_extend=x_extend.reshape(np.size(x_extend),1)

#extend y
y_extend=np.empty(len(x_extend)).reshape(np.size(x_extend),1)
y_extend[:]=np.nan

#merge x and y
data_extend = np.concatenate((x_extend,y_extend),axis=1)
data_extend = pd.DataFrame(data_extend, columns = ['x','y'])
data_extend.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.566371</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.591554</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.616737</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.641920</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.667103</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#concat original data with synthetic data
df=pd.concat([data, data_extend])
d=df.interpolate(method='linear', limit_direction ='both')
xx=np.array(d['x']).reshape(np.size(d['x']),1)
yy=np.array(d['y'])

#vizualise the data
plt.figure()
plt.title('linear interpolation')
plt.scatter(xx[:, 0], yy,alpha=0.3,label='data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
"""
Here we apply a linear interpolation, but the results aren't good for our data, 
so we decide to go with polynomial/cubic/spline...Etc
"""
```


![png](output_10_0.png)





    "\nHere we apply a linear interpolation, but the results aren't good for our data, \nso we decide to go with polynomial/cubic/spline...Etc\n"




```python
#concat original data with synthetic data
df=pd.concat([data, data_extend])
d=df.interpolate(method='polynomial', order=3, limit_direction ='both')
xx=np.array(d['x']).reshape(np.size(d['x']),1)
yy=np.array(d['y'])

#vizualise the data
plt.figure()
plt.title('poly interpolation')
plt.scatter(xx[:, 0], yy,alpha=0.3,label='data')
plt.xlabel('x')
plt.ylabel('y')
plt.show() 

""" 
The polynomial/cubic and other methods are just giving the same result, they clone the first data occurences
This what makes index methods and inerpolations weak
"""
```


![png](output_11_0.png)





    ' \nThe polynomial/cubic and other methods are just giving the same result, they clone the first data occurences\nThis what makes index methods and inerpolations weak\n'




```python
#We try to search for the period of our data distribution
d=df[370:500]
xx=np.array(d['x']).reshape(np.size(d['x']),1)
yy=np.array(d['y'])
plt.figure()
plt.scatter(xx[:, 0], yy,alpha=0.3,label='data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
"""
We can notice that there is a period of 130 element regulary, the idea is to save the steps (jumps) of the period elements
to clone it at the end of our dataset
"""
```


![png](output_12_0.png)





    '\nWe can notice that there is a period of 130 element regulary, the idea is to save the steps (jumps) of the period elements\nto clone it at the end of our dataset\n'




```python
#we init the start point as the last filled point
start = yy[-1]

#we compute all the differences
steps=np.diff(yy.ravel())[:len(data_extend)]

#we create a vector that will contain generated y data points
y_=np.zeros(len(data_extend))
y_[0]=data.iloc[-1]['y']



```


```python
for i in range(1,len(steps)):
    y_[i]=y_[i-1] + steps[i-1]
```


```python
data_extend['y']=y_
```


```python
#concat original data with synthetic data
df=pd.concat([data, data_extend])
#df=df.interpolate(method ='cubic', limit_direction ='backward')
#interpolate(method='index', inplace=True)
xx=np.array(df['x']).reshape(np.size(df['x']),1)
yy=np.array(df['y'])
plt.figure()
plt.title('data synth')
plt.scatter(xx[:, 0], yy,alpha=0.3,label='data source',color='darkorange')
plt.scatter(xx[-len(y_):, 0], y_,alpha=0.3,label='data generated', color='green')
plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})

plt.xlabel('x')
plt.ylabel('y')
plt.show()

"""
Here we can vizualise the extension from 12.5 to 15, the data looks perfect for the sinuosidal approx
"""
```


![png](output_16_0.png)





    '\nHere we can vizualise the extension from 12.5 to 15, the data looks perfect for the sinuosidal approx\n'




```python
reg = KernelRidge(kernel='rbf')
reg.fit(xx, yy)
p=reg.predict(xx)
```


```python
plt.figure()
plt.title(reg.__class__.__name__)
plt.scatter(xx[:, 0], yy,alpha=0.3,label='data', color='darkorange')
plt.scatter(xx[-len(y_):, 0], y_,alpha=0.3,label='data generated', color='green')
plt.plot(xx, p,color='blue',label='Kernel Ridge Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})
plt.show()  
```


![png](output_18_0.png)



```python

#Evaluation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

errorRMSE=mean_squared_error(yy,p)**0.5
errorMAE=mean_absolute_error(yy,p)**0.5

print(f'error RMSE= {errorRMSE} / erreur MAE= {errorMAE}')
```

    error RMSE= 0.6830907260219251 / erreur MAE= 0.7196513663690689
    


```python
#Vizualise error interval
plt.figure()
plt.title(reg.__class__.__name__)
plt.plot(xx, p,color='blue',label='Kernel Ridge Regression')
plt.fill_between(xx[:, 0], p - errorMAE, p + errorMAE, color='darkorange',alpha=0.5,label='MAE interval')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})
plt.show()  
```


![png](output_20_0.png)



```python
#Vizualise error interval
plt.figure()
plt.title(reg.__class__.__name__)
plt.plot(xx, p,color='blue',label='Kernel Ridge Regression')
plt.fill_between(xx[:, 0], p - errorRMSE, p + errorRMSE, color='green',alpha=0.5,label='RMSE interval')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})
plt.show()  
```


![png](output_21_0.png)



```python
"""
We notice that there is no big difference between the MSE/MAE errors, we can say that our model can err the y value
with a 0.7 more or less +-(0.7) 
"""
```
