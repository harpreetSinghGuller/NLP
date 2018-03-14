library(readr)
library(tm)
library(qdap)
library(plyr)
library(dplyr)
library(data.table)



clean_text <- function(text){
  text <- tolower(text)
  text <- removePunctuation(text) #contraction is also removed
  text <- removeNumbers(text)
  text <- stripWhitespace(text)
  
  text <- bracketX(text)  # (e.g. “It’s (so) cool” becomes “It’s cool”)
  #replace_number() # (e.g. “2” becomes “two”)
  #replace_abbreviation() #(e.g. “Sr” becomes “Senior”)
  #replace_contraction() #(e.g. “shouldn’t” becomes “should not”)
  #replace_symbol() #(e.g. “$” becomes “dollar”)
  
  # weren't => werent
  stopwords_no_contraction <- removePunctuation(stopwords('en'))
  
  ### Remove stop words
  text <- removeWords(text, stopwords_no_contraction)
  
  # Remove extra space
  text <- gsub(' +',' ', text)
  
  # Remove spaces at beginning and ending
  text <- strip(text)
  
  return(text)
}

clean_corpus <- function(corpus){
  #corpus <- tm_map(corpus, clean_text) # This caused error when using DocumentTermMatrix()
  corpus <- tm_map(corpus, content_transformer(clean_text))
  return(corpus)
}

build_corpus <- function(data_input){
  train_comments <- VectorSource(train$comment_text)
  
  # Create Documents Term Matrix 
  print("Building Corpus...")
  comments_corpus <- VCorpus(train_comments)
  
  print("Cleaning Corpus...")
  comments_corpus <- clean_corpus(comments_corpus)
  return(comments_corpus)
}




