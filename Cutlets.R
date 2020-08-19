# Hypothesis testing
# Cutlets data

library(readr)
cutlets <- read_csv("C:/Users/Agnelo Christy/Desktop/Data Science!/Assignment 3 - Hypothesis Testing/Cutlets.csv")
View(cutlets)

# Assigning new variables x,y for Unit A, Unit B respectively 

x <- cutlets$`Unit A`
x

y <- cutlets$`Unit B`
y

# Finding mean and standard deviation of x and y
mean(x)
sd(x)

mean(y)
sd(y)

# n1=35, n2=35



# Assessing by plotting a boxplot
boxplot(x,y)
# Labelling X and Y axis, also giving a name to the boxplot
boxplot(x,y, xlab = "Cutlets", ylab = "Weights", las=1, main = " Weights of culets")

# Normality test
shapiro.test(x) #p-value > 0.05
shapiro.test(y) #p-value > 0.05
# Both variables are normally distributed

# Variance test
var.test(x,y) #p-value > 0.05
# Varianes are equal



# Conducting Two sample t-test

# Null hypothesis : H_0 : Diff in mean is equal to 0
# Alternative hypothesis : H_a : Diff in mean is not equal to 0

# Two sample t-test
t.test(x,y, mu=0, alternative = "two.sided", paired = F, confint=0.95)
# paired = F, since data is not generated out of same individuals, they are independent of each other 

# Confidence interval ( -0.09654633 to 0.20613490)
# p-value= 0.4723 > 0.05
# Accept null hypothesis.
# Hence, we can say there is no significant difference in the diameters of Cutlets between the two units.