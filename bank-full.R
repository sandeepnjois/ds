# Bank

library(readr)

# Importing file through import dataset
View(bank.full)
str(bank.full)
dim(bank.full)
sum(is.na(bank.full))
class(bank.full)

b <- bank.full

# Revalue categorical variable 
class(b$job)

b$job[b$job == "management"] = "1"
b$job[b$job == "technician"] = "2"
b$job[b$job == "entrepreneur"] = "3"
b$job[b$job == "blue-collar"] = "4"
b$job[b$job == "retired"] = "5"
b$job[b$job == "services"] = "6"
b$job[b$job == "unemployed"] = "7"
b$job[b$job == "admin."] = "8"
b$job[b$job == "self-employed"] = "9"
b$job[b$job == "unknown"] = "10"
b$job[b$job == "housemaid"] = "11"
b$job[b$job == "student"] = "12"
View(b)

b$job <- as.factor(b$job)
class(b$job)
levels(b$job)


str(b)
b$marital[b$marital == "married"] = "0"
b$marital[b$marital == "single"] = "1"
b$marital[b$marital == "divorced"] = "2"

b$marital <- as.factor(b$marital)
class(b$marital)
levels(b$marital)

str(b)

b$education[b$education == "primary"] = "0"
b$education[b$education == "secondary"] = "1"
b$education[b$education == "tertiary"] = "2"
b$education[b$education == "unknown"] = "3"

b$education <- as.factor(b$education)
class(b$education)
levels(b$education)

str(b)

b$default[b$default == "no"] = "0"
b$default[b$default == "yes"] = "1"

b$default <- as.factor(b$default)
class(b$default)
levels(b$default)

str(b)

b$housing[b$housing == "no"] = "0"
b$housing[b$housing == "yes"] = "1"

b$housing <- as.factor(b$housing)
class(b$housing)
levels(b$housing)

str(b)

b$loan[b$loan == "no"] = "0"
b$loan[b$loan == "yes"] = "1"

b$loan <- as.factor(b$loan)
class(b$loan)
levels(b$loan)

str(b)

b$contact[b$contact == "unknown"] = "0"
b$contact[b$contact == "cellular"] = "1"
b$contact[b$contact == "telephone"] = "2"

b$contact <- as.factor(b$contact)
class(b$contact)
levels(b$contact)

str(b)

table(b$month)

b$month[b$month == "jan"] = "1"
b$month[b$month == "feb"] = "2"
b$month[b$month == "mar"] = "3"
b$month[b$month == "apr"] = "4"
b$month[b$month == "may"] = "5"
b$month[b$month == "jun"] = "6"
b$month[b$month == "jul"] = "7"
b$month[b$month == "aug"] = "8"
b$month[b$month == "sep"] = "9"
b$month[b$month == "oct"] = "10"
b$month[b$month == "nov"] = "11"
b$month[b$month == "dec"] = "12"

b$month <- as.factor(b$month)
class(b$month)
levles(b$month)

str(b)

table(b$poutcome)

b$poutcome[b$poutcome == "failure"] = "0"
b$poutcome[b$poutcome == "other"] = "1"
b$poutcome[b$poutcome == "success"] = "2"
b$poutcome[b$poutcome == "unknown"] = "3"

b$poutcome <- as.factor(b$poutcome)
class(b$poutcome)
levels(b$poutcome)

str(b)

table(b$y)

b$y[b$y == "no"] = "0"
b$y[b$y == "yes"] = "1"

b$y <- as.factor(b$y)
class(b$y)
levels(b$y)

str(b)
summary(b)

# Performing logistic regression
model <- glm(y ~ age+job+marital+education+default+balance+housing+loan+contact+day+month+duration+campaign+pdays+previous+poutcome, b, family = "binomial")
summary(model)
