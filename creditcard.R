# Creditcard

library(readr)

creditcard <- read_csv(file.choose())
View(creditcard)
summary(creditcard)

# Removing index 
cc <- creditcard[-1]
View(cc)

sum(is.na(cc)) # No N.A values
dim(cc) 
str(cc)
class(cc)
attach(cc)





# Revalue categorical variable 

cc$card[cc$card == "yes"] = "1"
cc$card[cc$card == "no"] = "0"

cc$owner[cc$owner=="yes"] = "1"
cc$owner[cc$owner=="no"] = "0"

cc$selfemp[cc$selfemp == "yes"] ="1"
cc$selfemp[cc$selfemp == "no"] = "0"
View(cc)
class(card)

cc$card <- as.factor(cc$card)
cc$owner <- as.factor(cc$owner)
cc$selfemp <- as.factor(cc$selfemp)
summary(cc)
str(cc)



# Performing logistic regression
model <- glm(card ~ reports+age+income+share+expenditure+owner+selfemp+dependents+months+majorcards+active, cc, family = "binomial")
summary(model)

