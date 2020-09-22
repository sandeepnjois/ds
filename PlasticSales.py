#Importing modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing datasset
ps = pd.read_csv("PlasticSales.csv")
months =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 

p = ps["Month"][0] #Assigning values of 'Quarter' in 'p'

ps['months']= 0

for i in range(60):
    p = ps["Month"][i]
    ps['months'][i]= p[0:3]
    
#EDA
#Sales
ps['Sales'].isnull().sum() # 0 N.A values
ps['Sales'].mean() 
ps['Sales'].median()
ps['Sales'].mode()
ps['Sales'].var()
ps['Sales'].std()

ps['Sales'].skew()
ps['Sales'].kurt()
#skewness = slight right skewed
#kurtosis = slight flat curve
plt.boxplot(ps['Sales'],1,'rs',0)
# 0 outliers found

#Preprocessing and encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

ps['months'] = label_encoder.fit_transform(ps['months'])
ps['months'] = ps['months'].astype('category')

#Creating new columns of 'TimeSeries'
#Creating a new variable 't'
ps["t"] = np.arange(1,61)
#Creating a new variable 't_squared'
ps["t_squared"] = ps["t"]*ps["t"]
#Creating a new variable 'log_Rider'
ps["log_Rider"] = np.log(ps["Sales"])

print(ps.columns)

#Dropping 'Quarter' column as it is a string type
ps = ps.drop('Month', axis =1)

#Splitting data into train and test data
Train = ps.head(48)
Test = ps.tail(12)

#Plot of 'Sales' variable 
plt.plot(ps.iloc[:,0])

## Linear ##
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear
#260.93

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
#297.41

## Additive seasonality ##
add_sea = smf.ols('Sales~months',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['months']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea
#235.60

## Additive Seasonality Quadratic ##
add_sea_Quad = smf.ols('Sales~t+t_squared+months',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['months','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 
#218.19

## Multiplicative Seasonality ##
Mul_sea = smf.ols('Sales~months',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea
#Inf

## Multiplicative Additive Seasonality ##
Mul_Add_sea = smf.ols('Sales~t+months',data = Train).fit()
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