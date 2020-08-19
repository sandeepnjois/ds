# Fantaloons dataset

library(readr)
fan <- read_csv(file.choose())
View(fan)
attach(fan)


# Stacking data
stacked_data <- stack(fan)
View(stacked_data)
attach(stacked_data)


# Stacking data
table1 <- table(values, ind)
table1



# 2-sample test for equality of proportions without continuity correction
prop.test(x=c(167,233), n=c(280,520), conf.level = 0.95, correct = FALSE, alternative = "two.sided")
# two.sided -> means checking for equal proportions
# p-value = 6.261e-05 < 0.05
# Reject to accept null hypothesis
# Proportions of male vs female walking into the store differ based on the day of the week