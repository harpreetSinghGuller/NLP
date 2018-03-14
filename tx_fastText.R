setwd("~/Google Drive/STUDY/NYC Data Science/Projects/tx_multilabel_class")
source("./data_process.R")

library(caret)
library(keras)
# Input Data processing

train <- read_csv("./input/train.csv")
submission <- read_csv("./input/test.csv")
# training Target class
labels <- colnames(train)[3:8]
targets <- train[,labels]

# Data Preparation --------------------------------------------------------
library(keras)
library(purrr)

# Parameters --------------------------------------------------------------

# ngram_range = 2 will add bi-grams features
ngram_range <- 2
max_features <- 20000
maxlen <- 400
batch_size <- 32
embedding_dims <- 50
epochs <- 5

# Function Definitions ----------------------------------------------------

create_ngram_set <- function(input_list, ngram_value = 2){
  indices <- map(0:(length(input_list) - ngram_value), ~1:ngram_value + .x)
  indices %>%
    map_chr(~input_list[.x] %>% paste(collapse = "|")) %>%
    unique()
}

add_ngram <- function(sequences, token_indice, ngram_range = 2){
  ngrams <- map(
    sequences, 
    create_ngram_set, ngram_value = ngram_range
  )
  
  seqs <- map2(sequences, ngrams, function(x, y){
    tokens <- token_indice$token[token_indice$ngrams %in% y]  
    c(x, tokens)
  })
  
  seqs
}


#######
train_text <- clean_text(train$comment_text)
# Building tokenizer basing on the toxic text
train_tx <- train %>% filter(toxic==1 | severe_toxic ==1 | obscene == 1 | 
                            threat == 1 | insult == 1 | identity_hate == 1) 
text_for_tokenizing <- clean_text(train_tx$comment_text)

tokenizer <- text_tokenizer(num_words = max_features)
fit_text_tokenizer(tokenizer, text_for_tokenizing)

# cleaning up memory and save tokenizer
rm(text_for_tokenizing, train_tx)
save_text_tokenizer(tokenizer, "tx_tokenizer")


# Building master text training set
trainset <- texts_to_sequences(tokenizer = tokenizer, train_text)

# Checking data
library(purrr)
print(length(trainset))
print(sprintf("Average train sequence length: %f", mean(map_int(trainset, length))))

# Working on sample of top 100
#trainset100 <- trainset[1:100]

# Building n-gram matrix
if(ngram_range > 1) {
  
  # Create set of unique n-gram from the training set.
  ngrams <- trainset %>% 
    map(create_ngram_set) %>%
    unlist() %>%
    unique()
  
  # Dictionary mapping n-gram token to a unique integer
  # Integer values are greater than max_features in order
  # to avoid collision with existing features
  token_indice <- data.frame(
    ngrams = ngrams,
    token  = 1:length(ngrams) + (max_features), 
    stringsAsFactors = FALSE
  )
  
  # max_features is the highest integer that could be found in the dataset
  max_features <- max(token_indice$token) + 1
  
  # Augmenting x_train and x_test with n-grams features
  trainset_ngram <- add_ngram(trainset, token_indice, ngram_range)
  #imdb_data$test$x <- add_ngram(imdb_data$test$x, token_indice, ngram_range)
}

#TRAINING PROCESSS ########################################
#catenum <- 2

# Build master training dataset
trainset_ngram <- pad_sequences(trainset, maxlen = maxlen)

#train_dataset1 <- data.frame(cbind(trainset_ngram, targets['toxic']))
#train_dataset <- data.frame(cbind(trainset_ngram, targets[catenum]))
train_dataset <- data.frame(cbind(trainset_ngram, targets))

## Splitting x_y_train and x_y_test datasets
train_dataset <- data.frame(cbind(trainset_ngram, train$toxic))

library(rsample)
set.seed(100)
train_test_split <- initial_split(train_dataset, prop = 0.8)
rm(train_dataset)

# Retrieve train and test sets
train_tbl <- training(train_test_split)
test_tbl  <- testing(train_test_split) 
rm(train_test_split)

# y_train <- train_tbl[,maxlen+1]
# x_train <- train_tbl[,-maxlen+1]
y_train <- train_tbl[,maxlen+c(1:6)]
x_train <- train_tbl[,-maxlen+c(1:6)]
x_train <- as.matrix(x_train)
rm(train_tbl)

# y_test <- test_tbl[,maxlen+1]
# x_test <- test_tbl[,-maxlen+1]
y_test <- test_tbl[,maxlen+c(1:6)]
x_test <- test_tbl[,-maxlen+c(1:6)]
x_test <- as.matrix(x_test)
rm(test_tbl)

y_train <- as.matrix(y_train)
y_test <- as.matrix(y_test)


# Model Definition --------------------------------------------------------
model <- keras_model_sequential()

model %>%
  layer_embedding(
    input_dim = max_features, output_dim = embedding_dims, 
    input_length = maxlen
  ) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(6, activation = "sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

# Fitting -----------------------------------------------------------------

model %>% fit(
  x_train, y_train, 
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2
)


# Saving model to file
model_filename <- paste0("model_fasttext_class_","ALL",".hdf5")
#model_filename <- paste0("model_fasttext_class_",labels[catenum],".hdf5")

save_model_hdf5(model, model_filename)

# Evaluating -----------------------------------------------------------------

best_model <- load_model_hdf5(model_filename)
score <- best_model %>% evaluate(
  x_test, y_test,
  batch_size = batch_size,
  verbose = 1
)
score

library(Metrics)
y_test_pred <- best_model %>% predict(x_test)

y_test_pred <- best_model %>% predict_classes(x_test)
result_test <- confusionMatrix(y_test_pred, y_test)
result_test

auc(y_test, y_test_pred)

library(pROC)
plot(roc(y_test, y_test_pred))

#### Submission
sub_text <- clean_text(submission$comment_text)

###
sub_ngram <- texts_to_sequences(tokenizer, sub_text)
sub_pad <- pad_sequences(sub_ngram, maxlen = maxlen)

sub_pred <- best_model %>% predict(sub_pad)
rm(sub_pad)

sub_data <- data.frame(submission$id,sub_pred)
colnames(sub_data) <- c("id",labels[1])
write.csv(sub_data, "submission.csv", row.names = FALSE)

### Appending new column
tmp <- read.csv("original_file.csv")
tmp <- cbind(tmp, new_column)
write.csv(tmp, "modified_file.csv")

###################<<<<<<<<<<<>>>>>>>>>>>####################
