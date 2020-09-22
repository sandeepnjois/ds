#Importing modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing datasset
cc = pd.read_excel("CocaCola_Sales_Rawdata.xlsx")
Qtr =['Q1','Q2','Q3','Q4'] #Creating a new list 'Qtr' 

p = cc["Quarter"][0] #Assigning values of 'Quarter' in 'p'

cc['Qtr']= 0

for i in range(42):
    p = cc["Quarter"][i]
    cc['Qtr'][i]= p[0:3]

#EDA
#Sales
cc['Sales'].isnull().sum() # 0 N.A values
cc['Sales'].mean() 
cc['Sales'].median()
cc['Sales'].mode()
cc['Sales'].var()
cc['Sales'].std()

cc['Sales'].skew()
cc['Sales'].kurt()
#skewness = slight right skewed
#kurtosis = slight flat curve
plt.boxplot(cc['Sales'],1,'rs',0)
# 0 outliers found

#Preprocessing and encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

cc['Qtr'] = label_encoder.fit_transform(cc['Qtr'])
cc['Qtr'] = cc['Qtr'].astype('category')

#Creating new columns of 'TimeSeries'
#Creating a new variable 't'
cc["t"] = np.arange(1,43)
#Creating a new variable 't_squared'
cc["t_squared"] = cc["t"]*cc["t"]
#Creating a new variable 'log_Rider'
cc["log_Rider"] = np.log(cc["Sales"])

print(cc.columns)

#Dropping 'Quarter' column as it is a string type
cc = cc.drop('Quarter', axis =1)

#Splitting data into train and test data
Train = cc.head(34)
Test = cc.tail(8)

#Plot of 'Sales' variable 
plt.plot(cc.iloc[:,0])

## Linear ##
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear
#720.61

## Exponential ##
Exp = smf.ols('Sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp
#Inf

## Quadratic ##
Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad
#437.74

## Additive seasonality ##
add_sea = smf.ols('Sales~Qtr',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Qtr']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea
#1870.19

## Additive Seasonality Quadratic ##
add_sea_Quad = smf.ols('Sales~t+t_squared+Qtr',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Qtr','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 
#257.67

## Multiplicative Seasonality ##
Mul_sea = smf.ols('Sales~Qtr',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea
#Inf

## Multiplicative Additive Seasonality ##
Mul_Add_sea = smf.ols('Sales~t+Qtr',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 
#Inf

### Testing ###
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
print(table_rmse)

# 'rmse_add_sea_quad' the least value among the models prepared so far 
# Only one dummy variable was created 'Qtr' using Encoding in every model
# So, we will using model built on ' Additive Seasonality Quadratic'