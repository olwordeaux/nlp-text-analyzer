# NLP Pipeline Script

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.6%2B-blue.svg)
![GitHub Issues](https://img.shields.io/github/issues/Olwordeaux/nlp-pipeline.svg)
![GitHub Stars](https://img.shields.io/github/stars/Olwordeaux/nlp-pipeline.svg)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Set Up Virtual Environment](#set-up-virtual-environment)
  - [Install Dependencies](#install-dependencies)
  - [Download NLTK Data](#download-nltk-data)
- [Usage](#usage)
  - [Prepare Your Text File](#prepare-your-text-file)
  - [Configure the Script](#configure-the-script)
  - [Run the Pipeline](#run-the-pipeline)
  - [Monitor Logs](#monitor-logs)
- [Dependencies](#dependencies)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

The **NLP Pipeline Script** is a comprehensive Natural Language Processing tool designed to analyze textual data efficiently. By leveraging powerful libraries such as NLTK, Gensim, and Scikit-learn, this pipeline performs the following key tasks:

- **Text Cleaning:** Removes unwanted characters, corrects spelling mistakes, and normalizes text.
- **Preprocessing:** Tokenizes text, removes stopwords, and filters non-alphabetic tokens.
- **Feature Extraction:** Utilizes TF-IDF to extract the top 1000 features.
- **Clustering:** Applies k-Means clustering to group similar text segments.
- **Topic Modeling:** Implements LDA (Latent Dirichlet Allocation) to identify underlying topics.
- **Keyword Extraction:** Identifies the most frequent keywords in the text.
- **Reporting:** Generates comprehensive reports summarizing the analysis.
- **Logging:** Maintains detailed logs for monitoring and debugging.

This pipeline is ideal for researchers, data scientists, and developers looking to perform in-depth text analysis with ease and reliability.

## Features

- **Robust Text Cleaning:** Eliminates digits, special characters, emails, URLs, and corrects spelling errors.
- **Advanced Preprocessing:** Efficient tokenization and stopword removal tailored for English texts.
- **TF-IDF Feature Extraction:** Extracts meaningful features to represent the text data numerically.
- **k-Means Clustering:** Groups similar text data into customizable clusters.
- **LDA Topic Modeling:** Discovers latent topics within the text for deeper insights.
- **Keyword Extraction:** Highlights the most significant words based on frequency.
- **Detailed Reporting:** Produces a comprehensive report summarizing all analysis steps and results.
- **Comprehensive Logging:** Keeps track of the pipeline's execution flow and errors for easy troubleshooting.

## Installation

Follow these steps to set up the NLP Pipeline Script on your local machine:

### Prerequisites

- **Operating System:** Windows, macOS, or Linux
- **Python:** Version 3.6 or higher
- **Git:** To clone the repository

### Clone the Repository

First, clone the repository from GitHub:

```bash
git clone https://github.com/Olwordeaux/nlp-pipeline.git
cd nlp-pipeline
Set Up Virtual Environment (Optional but Recommended)
Creating a virtual environment ensures that dependencies are managed separately from other projects.

python3 -m venv venv
Activate the virtual environment:

On macOS and Linux:

source venv/bin/activate
On Windows:

venv\Scripts\activate
Install Dependencies
Install the required Python libraries using pip:

pip install -r requirements.txt
Alternatively, install the libraries individually:

pip install nltk spellchecker gensim scikit-learn
Download NLTK Data
The script requires specific NLTK datasets. You can download them by running the following commands in a Python shell:

import nltk
nltk.download('punkt')
nltk.download('stopwords')
Alternatively, ensure these lines are included in your script as shown in the Usage section.

Usage
Follow these steps to execute the NLP pipeline:

Prepare Your Text File
Create or Obtain a Text File:

Ensure you have a .txt file containing the text you wish to analyze. For example, sample_text.txt.

Place the Text File:

Place your text file in the project directory or note its absolute path for configuration.

Configure the Script
Open the Script:

Open nlp_pipeline.py in your preferred text editor or IDE.

Set the File Path:

Modify the file_path variable to point to your text file:

file_path = 'sample_text.txt'  # Replace with your file path
For example, if your file is in a different directory:

file_path = '/path/to/your/your_text_file.txt'
Adjust Parameters (Optional):

You can customize parameters such as the number of clusters or topics by modifying the corresponding function calls at the end of the script.

Run the Pipeline
Execute the pipeline using Python:

python nlp_pipeline.py
If using a virtual environment, ensure it is activated before running the script.

Monitor Logs
The script logs its progress and any errors to nlp_pipeline.log. You can monitor this file in real-time using the following command:

On macOS and Linux:

tail -f nlp_pipeline.log
On Windows:

Use a text editor to open and view the log file.

Dependencies
The NLP Pipeline Script relies on the following Python libraries:

NLTK: Natural Language Toolkit for text processing.
SpellChecker: For correcting spelling mistakes.
Gensim: For topic modeling using LDA.
Scikit-learn: For TF-IDF feature extraction and k-Means clustering.
Others: Standard libraries such as re, logging, and collections are used for regular expressions, logging, and data handling respectively.
All dependencies are listed in the requirements.txt file for easy installation.

Outputs
Upon successful execution, the script generates the following outputs:

Log File (nlp_pipeline.log)
Description: Contains detailed logs of each step, including successful completions and any errors encountered.
Purpose: Useful for debugging and monitoring the pipeline's performance.
Location: Located in the project directory.
Report File (final_report_with_clustering_and_lda.txt)
Description: A comprehensive report summarizing the analysis.
Contents:
Original Text Preview: Shows the first 500 characters of the original text.
Cleaned Text Preview: Displays the first 500 characters of the cleaned text.
Filtered Tokens: Lists the first 100 preprocessed tokens.
Top 10 Keywords: Presents the most frequent keywords along with their occurrence counts.
k-Means Clusters: Details each cluster with its top contributing terms based on TF-IDF features.
LDA Topics: Lists the identified topics with their top associated words.
Purpose: Facilitates easy interpretation of the analysis results.
Location: Saved in the project directory with the name final_report_with_clustering_and_lda.txt.
Additional Outputs (Optional)
Visualization Files: If extended, the script can generate visualizations such as word clouds or cluster plots.
Enhanced Reports: Integration with formats like HTML or PDF for more polished reports.
Configuration
You can customize various aspects of the pipeline to suit your specific needs:

Adjusting Clustering Parameters
Number of Clusters:
kmeans = kmeans_clustering(tfidf_matrix, num_clusters=5)
Adjusting Topic Modeling Parameters
Number of Topics and Words:
lda_topics = lda_topic_modeling(processed_tokens, num_topics=5, num_words=10)
Output File Names
Report File:
generate_report(original_text, cleaned_text, processed_tokens, keywords, kmeans, feature_names, lda_topics, output_file='your_report_name.txt')
Logging Level
Change Logging Level:
logging.basicConfig(filename='nlp_pipeline.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
Options include DEBUG, INFO, WARNING, ERROR, and CRITICAL.

Contributing
Contributions are welcome! If you have suggestions, improvements, or bug fixes, please follow these steps:

Fork the Repository:
Click the "Fork" button on the GitHub repository to create your own copy.

Create a New Branch:
git checkout -b feature/YourFeatureName
Make Your Changes:
Implement your feature or fix in the codebase.

Commit Your Changes:
git commit -m "Add feature: YourFeatureName"
Push to Your Fork:
git push origin feature/YourFeatureName
Open a Pull Request:
Navigate to your fork on GitHub and click "Compare & pull request" to submit your changes for review.

Please ensure that your contributions adhere to the project's coding standards and include appropriate tests.

License
This project is licensed under the MIT License. You are free to use, modify, and distribute this software as per the terms of the license.
