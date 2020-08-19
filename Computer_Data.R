# Computer data

library(readr)

# Importing dataset

Comp <- read_csv("C:/Users/Agnelo Christy/Desktop/Data Science!/Assignment 5 - Multi Linear Regression/Computer_Data.csv")
View(Comp)
attach(Comp)

library(dplyr)

select(Comp,-X1)
comp <- select(Comp, -X1) # Removing the vairable X1 as it gives index values and has no influence on the output

View(comp)
attach(comp)


# Converting categorical variable in character type to factor type 
comp$multi <- as.factor(comp$multi)
comp$premium <- as.factor(comp$premium)
comp$cd <- as.factor(comp$cd)

attach(comp)
class(cd)
class(premium)
class(multi)


# Building a multiple linear regression model
model <- lm(price ~ speed+hd+ram+screen+cd+multi+premium+ads+trend)
summary(model)

# Final model

Final <- model
Final
# p-values of every variable is significant 
# R-squared value = 0.7756, shows good strength of the prediction model

confint(model, level = 0.95)
predict(model, intrerval="predict")
