#IMDB reviews on "FRIENDS" T.V show

king<-NULL
url1<-"https://www.imdb.com/title/tt0108778/reviews?ref_=tt_ql_3"

url<-read_html(as.character(paste(url1,sep="")))
ping<-url %>%
html_nodes(".show-more__control") %>% #UsingSelectorGadget
html_text() 
king<-c(king,ping)

write.table(king,"imdb_friends_reviews.txt") #Saving as a text file
getwd()

#Text cleaning
library(readr)
require(graphics)
library(tm)
library(SnowballC)
library(wordcloud)
library(RColorBrewer)
library(syuzhet)
library(ggplot2)


# Creating a corpus
text <- readLines(file.choose())
str(text)
View(text)

corpus <- Corpus(VectorSource(text)) # Converting dataframe into corpus
inspect(corpus[6])

# Clean text
removeURL <- function(x) gsub('http[[:alnum:]]*', '', x)
#Removing http or links
corpus <- tm_map(corpus, content_transformer(removeURL))
inspect(corpus[6])

corpus <- tm_map(corpus, tolower)
inspect(corpus[6])

corpus <- tm_map(corpus, removePunctuation)
inspect(corpus[6])

corpus <- tm_map(corpus, removeNumbers)
inspect(corpus[6])

corpus <- tm_map(corpus, removeWords, stopwords('english'))
inspect(corpus[6])

corpus <- tm_map(corpus, removeWords, c('way', 'just', 'ive', 'bit','let','yet','friends','show','watch'))
inspect(corpus[6])

corpus <- tm_map(corpus, stripWhitespace)
inspect(corpus[6])


writeLines(as.character(corpus), con="corpus_friends_imdb.txt")
#Saving the corpus file after text cleaning
getwd()

# Creating a document-term sparse matrix
tdm <- TermDocumentMatrix(corpus, control = list(minWordLength=c(2,8)))

findFreqTerms(tdm, lowfreq = 5)
#Words used 5 or more times
# 62 words are used 5 or more times

#Barplot
termFrequency <- rowSums(as.matrix(tdm))
termFrequency <- subset(termFrequency, termFrequency>=10)
#Frequency plot of words used 10 or more times
barplot(termFrequency,las=2, col = rainbow(20))
# 'Like', 'best' and 'love' are the most used words, which shows the show is good.

#Wordcloud
library(wordcloud)
m <- as.matrix(tdm)
wordFreq <- sort(rowSums(m), decreasing=TRUE)
wordcloud(words=names(wordFreq), freq=wordFreq, min.freq = 5, random.order = F, col=gray.colors(1))
wordcloud(words=names(wordFreq), freq=wordFreq, min.freq = 5, random.order = F, colors=rainbow(20))

#Sentiment analysis
#Regular sentiment score using get_sentiment() function and any method of choice

text <- readLines(file.choose())
View(text)
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
#Most positive scores

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
td_new <- data.frame(rowSums(td[1:15]))
#Transformation and cleaning
names(td_new)[1] <- "count"
td_new <- cbind("sentiment" = rownames(td_new), td_new)
rownames(td_new) <- NULL
td_new2<-td_new
#Plot One - count of words associated with each sentiment
quickplot(sentiment, data=td_new2, weight=count, geom="bar", fill=sentiment, ylab="count")+ggtitle("Frequency of words w.r.t Sentiments")
#Top emotions are 'positive','joy','trust','anticipation'

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
#Positive emotion is more than 2 times that of negative, concluding this 'TV show' is Good.