#Importing libraries for tweets extraction
library(twitteR)
library(ROAuth)
library(base64enc)
library(httpuv)

#Saving all API and credentials 
credentials <- OAuthFactory$new(consumerKey = "1WibEXCGFegtdTgk6BOCb",
                                consumerSecret = "K1qaoaBgHWmoD7evqPX2EG0k5twbTwNrpQC7lDFy3O0vBm",
                                requestURL = "https://api.twitter.com/oauth/request_token",
                                accessURL = "https://api.twitter.com/oauth/access_token",
                                authURL = "https://api.twitter.com/oauth/authorize")


#Saving credentials in a file
save(credentials, file = "Twitterauthenticationdata.R")
load("Twitterauthenticationdata.R")

setup_twitter_oauth("1WibEXCGFegtdTgk6BOCb", # Consumer Key (API Key)
                    "K1qaoaBgHWmoD7evqPX2EG0k5twbTwNrpQC7lDFy3O0vBm", #Consumer Secret (API Secret)
                    "1298848162087680-8ZfhEw6G0aUKiLl2kxJql1PMzJ3dCI",  # Access Token
                    "HfkeD4Jf6S6sDe5nCHotNFpxteaRRMs58VDjhFWt")  #Access Token Secret

#As API and keys from twitter dev app is not shareable. Keys are altered here after scraping the data. 

#Scrapping tweets from user ' Virat Kohli '
tweets <- userTimeline("imVkohli", n = 500)
tweetsDF <- twListToDF(tweets)
dim(tweetsDF)
View(tweetsDF)

#Saving tweets in .csv format
write.csv(tweetsDF, "newtweets_vk.csv", row.names = F)
#Directory
getwd()

#Importing necessary libraries for Sentiment analysis
library(readr)
require(graphics)
library(tm)
library(SnowballC)
library(wordcloud)
library(RColorBrewer)
library(syuzhet)
library(ggplot2)


# Creating a corpus
tweets_vk <- read.csv(file.choose(), header =T)
str(tweets_vk)
corpus <- tweets_vk$text #We need only the text column of the dataframe
corpus <- Corpus(VectorSource(corpus)) # Converting dataframe into corpus
inspect(corpus[1:5])

# Clean text
removeURL <- function(x) gsub('http[[:alnum:]]*', '', x)
corpus <- tm_map(corpus, content_transformer(removeURL))
inspect(corpus[1:5])

corpus <- tm_map(corpus, tolower)
inspect(corpus[1:5])

corpus <- tm_map(corpus, removePunctuation)
inspect(corpus[1:5])

corpus <- tm_map(corpus, removeNumbers)
inspect(corpus[1:5])

corpus <- tm_map(corpus, removeWords, stopwords('english'))
inspect(corpus[1:5])

cleanset <- tm_map(corpus, removeWords, c('ufeeuff', 'ufufaa','ufeufd','ufa','cwc','ufc'))
inspect(corpus[1:5])

corpus <- tm_map(corpus, stripWhitespace)
inspect(corpus[1:5])

corpus <- tm_map(corpus, gsub, 
                   pattern = 'thanks', 
                   replacement = 'thank')
#All 'thanks' word will be considered as 'thank'

writeLines(as.character(corpus), con="newcorpus_vk.txt")
#Saving the corpus file after text cleaning
getwd()

# Creating a document-term sparse matrix
tdm <- TermDocumentMatrix(corpus, control = list(minWordLength=c(2,8)))

findFreqTerms(tdm, lowfreq = 5)
#Words used 5 or more times

#Barplot
termFrequency <- rowSums(as.matrix(tdm))
termFrequency <- subset(termFrequency, termFrequency>=10)
#Frequency plot of words used 10 or more times
barplot(termFrequency,las=2, col = rainbow(20))
#'Thank' and 'happy' are most used words

#Wordcloud
library(wordcloud)
m <- as.matrix(tdm)
wordFreq <- sort(rowSums(m), decreasing=TRUE)
wordcloud(words=names(wordFreq), freq=wordFreq, min.freq = 5, random.order = F, col=gray.colors(1))
wordcloud(words=names(wordFreq), freq=wordFreq, min.freq = 5, random.order = F, colors=rainbow(20))

#Sentiment analysis
#Regular sentiment score using get_sentiment() function and any method of choice

text <- readLines(file.choose())

# 'syuzhet' method
syuzhet_vector <- get_sentiment(text, method="syuzhet")
#First row of the vector
head(syuzhet_vector)
#Smmary statistics of the vector
summary(syuzhet_vector)

# 'bing' method
bing_vector <- get_sentiment(text, method="bing")
head(bing_vector)
summary(bing_vector)

# 'affin' method
afinn_vector <- get_sentiment(text, method="afinn")
head(afinn_vector)
summary(afinn_vector)

#Comparing the first row of each vector using sign function
rbind(
  sign(head(syuzhet_vector)),
  sign(head(bing_vector)),
  sign(head(afinn_vector))
)

#All positive scores


#Run nrc sentiment analysis to return data frame with each row classified as one of the following
#Emotions, rather than a score: 
#Anger, anticipation, disgust, fear, joy, sadness, surprise, trust
#It also counts the number of positive and negative emotions found in each row
d<-get_nrc_sentiment(text)
# head(d,10) - to see top 10 lines of the get_nrc_sentiment dataframe
head (d,10)

#Transpose
td<-data.frame(t(d))
#The function rowSums computes column sums across rows for each level of a grouping variable.
td_new <- data.frame(rowSums(td[1:50]))
#Transformation and cleaning
names(td_new)[1] <- "count"
td_new <- cbind("sentiment" = rownames(td_new), td_new)
rownames(td_new) <- NULL
td_new2<-td_new
#Plot One - count of words associated with each sentiment
quickplot(sentiment, data=td_new2, weight=count, geom="bar", fill=sentiment, ylab="count")+ggtitle("Frequency of words w.r.t Sentiments")
#'Positive','joy','trust','anticipation,'surprise' are the top conveyed emotions
#'
#Plot two - count of words associated with each sentiment, expressed as a percentage
barplot(
  sort(colSums(prop.table(d[, 1:10]))), 
  horiz = TRUE, 
  cex.names = 0.7, 
  las = 1, 
  main = "Emotions in Text", xlab="Percentage"
)

#Plot of count of words associated with positive and negative emotions
td_pos_neg <-  td_new[9:10,]
quickplot(sentiment, data=td_pos_neg, weight=count, geom="bar", fill=sentiment, ylab="count")+ggtitle("Positive and negative emotions")
#Positive tweets are way more than negative tweets