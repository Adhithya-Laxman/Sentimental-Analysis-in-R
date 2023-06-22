# Load the required libraries
library(shiny)
library(ggplot2)

# Define the UI
ui <- fluidPage(
  titlePanel("Sentiment Analysis"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Choose CSV File"),
      checkboxInput("removeStopwords", "Remove Stopwords", value = TRUE),
      actionButton("runAnalysis", "Run Analysis"),
      
    ),
    mainPanel(
      plotOutput("emotionPlot"),
      tableOutput("confusionMatrix"),
      tableOutput("metrics"),
      tableOutput("outputTable")
      
    )
  )
)

# Define the server
server <- function(input, output) {
  # Load the CSV file
  data <- reactive({
    req(input$file)
    read.csv(input$file$datapath)
  })
  
  output_df <- data.frame(Line = character(), Sentiment = character(), stringsAsFactors = FALSE)
  
  
  
  # Perform sentiment analysis
  observeEvent(input$runAnalysis, {
    # Preprocessing code here
    library(caret)
    
    df  =data()
    
    df = subset(df, df$sentiment=='positive' | df$sentiment=='negative')
    
    text_column <- df[, 1]
    
    # Remove punctuation and other special characters
    text_column <- gsub("[^a-zA-Z0-9 ]", "", text_column)
    
    # Remove leading and trailing whitespaces
    text_column <- trimws(text_column)
    
    # Remove stopwords (optional)
    # You can use the stopwords provided by the 'tm' package or customize your own list
    library(tm)
    text_column <- removeWords(text_column, stopwords("english"))
    text_column <- tolower(text_column)
    # Update the cleaned text back to the data frame
    df[, 2] <- text_column
    library(Matrix)
    
    set.seed(123)
    
    # Split the data into training, testing, and verification datasets
    train_indices <- sample(1:nrow(df), 0.7 * nrow(df))  # 60% for training
    test_indices <- sample(setdiff(1:nrow(df), train_indices), 0.2 * nrow(df))  # 30% for testing
    verify_indices <- setdiff(1:nrow(df), c(train_indices, test_indices))  # remaining data for verification
    
    # Create the training, testing, and verification datasets
    train_data <- df[train_indices, ]
    test_data <- df[test_indices, ]
    verify_data <- df[verify_indices, ]
    
    train_data
    # Write processed clean text into cleaned_text
    cleaned_text <- train_data[,1]
    
    # Create a corpus from the text data
    corpus <- Corpus(VectorSource(cleaned_text))
    
    # Preprocess the corpus
    corpus <- tm_map(corpus, content_transformer(tolower))
    corpus <- tm_map(corpus, removePunctuation)
    corpus <- tm_map(corpus, removeNumbers)
    corpus <- tm_map(corpus, removeWords, stopwords("english"))
    
    # Create a document-term matrix
    dtm <- DocumentTermMatrix(corpus)
    
    
    # Calculate TF-IDF
    tfidf <- weightTfIdf(dtm)
    
    library(tidytext)
    
    sentiments = get_sentiments()
    
    pos = subset(sentiments,sentiments$sentiment=='positive')
    
    pos = pos[,1]
    
    pos
    
    neg = subset(sentiments,sentiments$sentiment=='negative')
    neg
    neg = neg[,1]
    
    neg
    
    
    
    labels_df <- data.frame(
      Emotion = c(
        "positive", "negative"
      ),
      Words = c(
        paste(as.character(pos),"happy, positive,fantastic, anticipation, eager, exciting,excite, hopeful, yearning, hype, can't wait, countdown, anticipation level: 100, on the edge of my seat, thrilled, anxious,joy, happy, delighted, ecstatic, blissful, lol, happy dance, cheers, ROFL, high-five, excited, thrilled, like, good, love, haha,surprise, amazed, astonished, shocked, startled, OMG, mind blown, plot twist, jaw-dropping, wowza, shocked, stunned, new, better,trust, trusted, confident, faithful, reliable, BFF, loyal, reliable source, confidante, trust fall, reliable, trustworthy, honest, dependable, work,love, like, positive, success, happiness, love, thumbs up, amazing, awesome sauce, winning, happy, grateful, optimistic, blessed, loved "),
        paste(as.character(neg),"sad, negative, disgust, disgusted,not,please loathing, revulsion, repelled,suck, eww, grossed out, vomit, cringe, sickened, gross, nauseated, revolted, repulsed.anger, angry, annoyed, furious, rage, ragequit, triggered, salty, hater, flame, hate, mad, irritatedfear, afraid, scared, terrified, panic, spooky, scaredy-cat, nightmare, goosebumps, chilling, anxious, worried sadness, sad, depressed, sorrow, grief, heartbroken, tears, down in the dumps, lonely, sob, miserable, gloomy, miss,fail, hate, dislike, negative, failure, disappointment, haters gonna hate, fail, facepalm, disappointed, nope, frustrated, annoyed, jealous, regretful, guilty, love, well,  cold, bored, miss, sad, sick, sorry, tired, not, ugh, no, little, doesn't, never, sucks")
      ),
      stringsAsFactors = FALSE
    )
    
    
    # Install the 'naivebayes' package if not already installed
    # install.packages("naivebayes")
    # Load the 'naivebayes' library
    library(e1071)
    
    
    # Convert the TF-IDF matrix to a data frame
    tfidf_df <- as.data.frame(as.matrix(tfidf))
    
    # Repeat the sentiment labels to match the number of rows in tfidf_df
    num_rows <- nrow(tfidf_df)
    labels <- rep(labels_df$Emotion, length.out = num_rows)
    
    # Add the sentiment labels to the TF-IDF data frame
    tfidf_df$Sentiment <- labels
    
    # Train the Naive Bayes classifier
    naive_model <- naiveBayes(Sentiment ~ ., data = tfidf_df,laplace = 1)
    
    # Assuming you have a test dataset called 'test_data' with preprocessed text in the 6th column
    test_text <- test_data[, 1]
    test_text
    # Preprocess the test text using the same steps as before
    # TESTING
    test_text <- tolower(test_text)
    test_text <- gsub("[^a-zA-Z0-9 ]", "", test_text)
    test_text <- trimws(test_text)
    test_text <- removeWords(test_text, stopwords("english"))
    
    # Convert the test text to a document-term matrix using the previous corpus
    test_corpus <- Corpus(VectorSource(test_text))
    test_corpus <- tm_map(test_corpus, content_transformer(tolower))
    test_corpus <- tm_map(test_corpus, removePunctuation)
    test_corpus <- tm_map(test_corpus, removeNumbers)
    test_corpus <- tm_map(test_corpus, removeWords, stopwords("english"))
    
    test_dtm <- DocumentTermMatrix(test_corpus)
    
    # Calculate TF-IDF for the test data using the previous term frequency matrix
    test_tfidf <- weightTfIdf(test_dtm)
    
    # Convert the TF-IDF matrix to a data frame
    test_tfidf_df <- as.data.frame(as.matrix(test_tfidf))
    
    # Use the trained Naive Bayes model to predict sentiment labels for the test data
    predicted_sentiment <- predict(naive_model, newdata = test_tfidf_df)
    
    
    
    test_data <- na.omit(test_data)
    
    c = 0
    # n = nrows(predicted_sentiment)
    predicted_sentiment
    
    # while()
    # Create a data frame with the Line and Sentiment columns
    
    result_df <- test_data[predicted_sentiment %in% labels_df$Emotion, c("Line")]
    
    result_df
    print(result_df)
    
    
    for (i in 1:100) {
      line <- test_data[i, 1]
      l2 <- test_data[i, 3]
      sentiment <- predicted_sentiment[i]
      output_df <- rbind(output_df, data.frame(Line = line, Sentiment = sentiment, stringsAsFactors = FALSE))
    }
    
    library(ggplot2)
    
    # Count the occurrences of each sentiment
    sentiment_counts <- table(predicted_sentiment)
    
    # Create a data frame from the sentiment counts
    sentiment_data <- data.frame(Sentiment = names(sentiment_counts),
                                 Count = as.numeric(sentiment_counts))
    
    actual_sentiments <- test_data[,3]
    
    # Plot emotions distribution
    output$emotionPlot <- renderPlot({
      sentiment_counts <- table(predicted_sentiment)
      sentiment_data <- data.frame(Sentiment = names(sentiment_counts),
                                   Count = as.numeric(sentiment_counts))
      ggplot(sentiment_data, aes(x = Sentiment, y = Count)) +
        geom_bar(stat = "identity", fill = "steelblue") +
        labs(x = "Sentiment", y = "Count") +
        ggtitle("Emotions Distribution") +
        theme_minimal()
    })
    # CONFUSION MATRIX
    
    confusion_matrix <- confusionMatrix(data = factor(predicted_sentiment),
                                        reference = factor(actual_sentiments))
    
    # Extract the confusion matrix values
    true_negatives <- confusion_matrix$table[1, 1]
    false_negatives <- confusion_matrix$table[2, 1]
    true_positives <- confusion_matrix$table[2, 2]
    false_positives <- confusion_matrix$table[1, 2]
    
    # Calculate accuracy
    accuracy <- sum(diag(confusion_matrix$table)) / sum(confusion_matrix$table)
    
    # Calculate sensitivity (true positive rate)
    sensitivity <- true_positives / (true_positives + false_negatives)
    
    # Calculate specificity (true negative rate)
    specificity <- true_negatives / (true_negatives + false_positives)
    
    # Calculate precision (positive predictive value)
    precision <- true_positives / (true_positives + false_positives)
    
    # Calculate F1 score
    f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)
    
    # Create a data frame with the metrics
    metrics <- data.frame(
      Metric = c("Accuracy", "Sensitivity", "Specificity", "Precision", "F1 Score"),
      Value = c(accuracy, sensitivity, specificity, precision, f1_score)
    )
    metrics_table <- reactive({
      metrics
    })
    
    
    
    # Extract values from confusion matrix
    confusion_values <- as.data.frame(confusion_matrix$table)
    confusion_values$Metric <- rownames(confusion_values)
    
    
    # Display the confusion matrix
    output$confusionMatrix <- renderTable({
      confusion_values
    })
    
    output$metrics <- renderTable({
      metrics_table()
    })
    output$outputTable <- renderTable({
      output_df
    })
    
  })
  
}

# Run the Shiny app
shinyApp(ui, server)