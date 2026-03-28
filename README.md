Here is the adjusted `README.md` formatted specifically for a Google Colab environment. I updated the setup instructions to match how your code handles file paths and execution in Colab.

***

# Fake News Detection Pipeline

An end-to-end machine learning project developed in Google Colab that leverages both unsupervised and supervised algorithms to classify news articles as real or fake. 

## Overview
This repository contains a complete pipeline for processing text data and predicting the authenticity of news articles. It utilizes a Kaggle dataset of news articles. The project explores the data structure using K-Means clustering and PCA visualization before training a robust Random Forest classifier for final predictions.

## Features
* **Automated Data Handling:** Extracts data from zip archives directly within the Colab environment and loads it into a Pandas DataFrame.
* **Text Preprocessing:** Cleans article text by converting it to lowercase and stripping punctuation using regular expressions.
* **Feature Engineering:** Transforms raw text into numerical data using TF-IDF Vectorization, capped at 3,000 maximum features.
* **Unsupervised Learning:** Applies K-Means clustering (2 clusters) to discover latent topics and patterns without labeled data, visualized using 2D Principal Component Analysis (PCA).
* **Supervised Classification:** Trains a Random Forest Classifier with 200 estimators and balanced class weights to address dataset asymmetry, achieving an overall accuracy of 77%.

## Tech Stack
* **Environment:** Google Colab
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (KMeans, PCA, RandomForestClassifier, TfidfVectorizer)
* **Visualization:** Matplotlib, Seaborn

## Dataset
The model is trained on a `.csv` dataset sourced from Kaggle containing news articles, their authors, publication details, and binary labels indicating whether the content is real or fake. The pipeline dynamically maps label variations (e.g., "Fake"/"FAKE" to `0` and "Real"/"REAL" to `1`) and drops rows with missing text or labels.

## Model Performance
The Random Forest classifier achieved the following performance on the test set (20% split):

* **Overall Accuracy:** 77%
* **Fake News (Class 0):** * Precision: 0.75
  * Recall: 0.93
  * F1-Score: 0.83
* **Real News (Class 1):** * Precision: 0.83
  * Recall: 0.53
  * F1-Score: 0.64

## How to Run in Google Colab

Since this project was built for Google Colab, no local installation is required. Most dependencies (Pandas, Scikit-Learn, Matplotlib, Seaborn) are pre-installed in the Colab environment.

1. **Open Google Colab:**
   Navigate to [Google Colab](https://colab.research.google.com/) and upload the `.ipynb` notebook file from this repository.

2. **Upload the Dataset:**
   * Download the `news_articles.csv.zip` dataset.
   * In your Colab notebook, click on the **Files** icon (folder symbol) on the left sidebar.
   * Click the **Upload** icon and select your `news_articles.csv.zip` file.
   * *Note: Ensure the uploaded file is in the root `/content/` directory so the script can locate it.*

3. **Run the Notebook:**
   * Go to the top menu and select **Runtime** > **Run all** (or press `Ctrl + F9`).
   * The notebook will automatically unzip the dataset, clean the data, display the PCA clustering visualization, and output the final Random Forest classification report.
