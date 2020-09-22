#Importing modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing datasset
ad = pd.read_excel("Airlines_Data.xlsx")
months =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec',
         'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec',
         'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec',
         'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec',
         'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec',
         'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec',
         'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec',
         'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
# New column of months for 8 years

ad['months'] = months #Adding 'months' column
ad = ad.drop('Month',axis =1) #Dropping 'Month' column with 'TimeStamp'

#EDA
#Sales
ad['Passengers'].isnull().sum() # 0 NA values
ad['Passengers'].mean() 
ad['Passengers'].median()
ad['Passengers'].mode()
ad['Passengers'].var()
ad['Passengers'].std()

ad['Passengers'].skew() 
ad['Passengers'].kurt() 
#skewness = moderately right skewed
#kurtosis = slight flat curve
plt.boxplot(ad['Passengers'],1,'rs',0)
# 0 outliers found

# Label encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

# Assigning dummy values 
ad['months'] = label_encoder.fit_transform(ad['months'])
ad['months'] = ad['months'].astype('category')
ad['months'].unique()
type('Month')

#Creating new columns of 'TimeSeries'
#Creating a new variable 't'
ad["t"] = np.arange(1,97)
#Creating a new variable 't_squared'
ad["t_squared"] = ad["t"]*ad["t"]
#Creating a new variable 'log_Rider'
ad["log_Rider"] = np.log(ad["Passengers"])

print(ad.columns)

#Splitting data into train and test data
Train = ad.head(84)
Test = ad.tail(12)

plt.plot(ad.iloc[:,0])

##Forecasting models##
## Linear ##
import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear
#53.19

## Exponential ##
Exp = smf.ols('Passengers~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp
#9.115349885364664e+134

## Quadratic ##
Quad = smf.ols('Passengers~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad
#48.05

## Additive seasonality ##
add_sea = smf.ols('Passengers~months',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['months']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea
#132.82

## Additive Seasonality Quadratic ##
add_sea_Quad = smf.ols('Passengers~t+t_squared+months',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['months','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 
#26.36

## Multiplicative Seasonality ##
Mul_sea = smf.ols('Passengers~months',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea
#3.474696027791289e+103

## Multiplicative Additive Seasonality ##
Mul_Add_sea = smf.ols('Passengers~t+months',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 
#3.484154611152932e+148

### Testing ###
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea "]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
print(table_rmse)

# 'rmse_add_sea_quad' has the least value among the models prepared so far 
# Only one dummy variable was created 'months' using Encoding in every model
# So, we will using model built on ' Additive Seasonality Quadratic'