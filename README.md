# Sentiment Analysis in R

This project implements sentiment analysis using a Naive Bayes classifier in R. The application is built using the Shiny framework for interactive web applications.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Features](#features)
- [Code Overview](#code-overview)
- [Contributing](#contributing)

## Introduction

Sentiment analysis is a natural language processing (NLP) task that involves determining the sentiment expressed in a piece of text, such as positive, negative, or neutral. This project focuses on sentiment analysis from textual data using a Naive Bayes classifier.

## Getting Started

To run the project locally, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Adhithya-Laxman/Sentimental-Analysis-in-R.git
   cd Sentimental-Analysis-in-R
   ```

2. **Install Dependencies:**
   Ensure you have the required R libraries installed. You can install them using:
   ```R
   install.packages(c("shiny", "ggplot2", "caret", "tm", "e1071", "tidytext"))
   ```

3. **Run the Application:**
   Open the R script `app.R` in RStudio or your preferred R environment and run the application.

4. **Use the Application:**
   Open your web browser and navigate to `http://127.0.0.1:port`, where `port` is the port specified in your R environment.

## Features

- **CSV Input:**
  - Users can upload a CSV file containing text data for sentiment analysis.

- **Remove Stopwords:**
  - An option to remove stopwords from the text data for more accurate analysis.

- **Run Analysis:**
  - Perform sentiment analysis on the uploaded data with the click of a button.

- **Emotion Plot:**
  - Visualize the distribution of sentiments in the data through an interactive plot.

- **Confusion Matrix:**
  - Evaluate the performance of the sentiment analysis with a confusion matrix.

- **Metrics:**
  - Display accuracy, sensitivity, specificity, precision, and F1 score as evaluation metrics.

- **Output Table:**
  - View a table containing sampled lines and their predicted sentiments.

## Code Overview

The code is organized into the following sections:

- **UI Definition:**
  - The Shiny UI is defined in the `ui` variable, specifying the layout and components of the web application.

- **Server Logic:**
  - The server logic is defined in the `server` function, including data loading, preprocessing, model training, and result visualization.

- **Sentiment Analysis:**
  - The Naive Bayes classifier is trained on the provided data, and sentiment analysis is performed on a test dataset.

- **Visualization:**
  - The project includes visualization components such as the emotion plot, confusion matrix, and evaluation metrics.

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, feel free to open an issue or submit a pull request.


```

