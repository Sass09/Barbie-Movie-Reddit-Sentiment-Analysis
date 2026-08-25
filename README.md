# 🎀 Barbie Reddit Sentiment Analysis

An NLP and sentiment analysis project exploring how Reddit users responded to the 2023 *Barbie* movie.

🔗 **[View the Live Streamlit Dashboard](https://barbie-movie-reddit-sentiment-analysisgit-7iqytvuxmz5hwzsr8nob.streamlit.app)**

## Background and Overview

The *Barbie* movie was released in 2023 as the first live-action film centered on Barbie and her world alongside Ken.T here was a lot of commentary on the movie's plot, its creative direction and the social commentary involved in it. Given my interest in sentiment analysis, I decided to use VADER was an  easier tool to work with as a beginner as well as allowed me  to understand the outputs. Secondly, as a lifelong Barbie doll fan, I naturally gravitated towards this movie for my first foray in sentiment analysis project.

As someone who enjoys reading about people's thougts on different subject matters, Reddit stood out as a useful and my go-to source for exploring these reactions because discussions are often detailed, conversational, and shaped by community engagement through upvotes and replies.

This project uses Natural Language Processing (NLP), sentiment analysis, and data visualization to explore Reddit discussions surrounding the movie and answer questions such as:

* What was the overall sentiment of the discussion?
* What words and phrases appeared most frequently?
* What were some of the most strongly positive and negative reactions?
* Did longer comments receive more community engagement?

---

## 📊 Dashboard

The analysis was developed into an interactive dashboard using **Streamlit**.

The dashboard includes:

* Summary metrics for sentiment and engagement
* Overall sentiment distribution
* Barbie-shaped word cloud
* Most frequently occurring bigrams
* Comment length vs. upvotes
* Most upvoted strongly positive comments
* Most upvoted strongly negative comments



---

## 📥 Data Collection

Reddit data was collected using **PRAW (Python Reddit API Wrapper)**.

PRAW allows Python applications to access Reddit data through Reddit's API using credentials associated with a registered Reddit application.

Comments from relevant Reddit discussions about *Barbie* were collected along with engagement information such as upvotes.

---

## 🧹 Text Preprocessing

Before conducting the text analysis, Reddit comments were cleaned and standardized using a custom Python preprocessing function.

The text_preprocessing function was inspired by the original function used for this project  Model Deployment w/ Streamlit | Twitter (X.com) Sentiment Analysis. Found HERE.https://supertype.ai/notes/twitter-sentiment-analysis-part-3 

The preprocessing pipeline:

* Converted text to lowercase
* Replaced negative contractions with `"not"` to preserve negation
* Removed URLs
* Removed Reddit usernames
* Removed HTML entities and non-alphabetic characters
* Tokenized comments using NLTK
* Removed stopwords and single-character tokens
* Applied part-of-speech (POS) tagging
* Lemmatized words using WordNet based on their POS tags

The resulting tokens were used for analyses such as word frequency, bigrams, and the Barbie-shaped word cloud.

---

## 💬 Sentiment Analysis

Sentiment analysis was performed using **VADER (Valence Aware Dictionary and sEntiment Reasoner)**. Text Blot was also used on the text but not included for the any of the dashboard metrics and visualization. 

VADER produces sentiment scores representing the positive, neutral, and negative components of a piece of text, along with a **compound score** representing its overall sentiment.

During exploratory analysis, I found that relying solely on VADER classifications did not always identify comments that appeared meaningfully positive or negative when read in context.

For this reason, I decided to add additional filtering criteria were used when identifying comments for the strongly positive and negative comment tables using thresholds. 

### Strongly Positive Comments

I considered comment as considered strongly positive when:

* Its VADER compound score was greater than **0.7**
* Its upvote count was at or above the **80th percentile**

Including engagement helped surface strongly positive comments that also resonated with the Reddit community.

### Strongly Negative Comments

Strongly negative comments were filtered using stricter VADER criteria:

* Compound sentiment score below **-0.8**
* Neutral score of at least **0.7**

The resulting comments were then ranked by upvotes to identify highly engaged negative reactions.

---

## 🔤 Word and Bigram Analysis

After preprocessing, cleaned tokens from the comments were combined to examine frequently occurring language.

### Word Cloud

The exciting part was the custom Barbie-shaped word cloud which was generated to visualize frequently occurring words in the Reddit discussion. 
Firstly I downloaded the black and white image as a mask. The mask was then converted into pixels which gave wordcloud a clear area to populate.
### Bigrams

Bigrams are pairs of words that occur consecutively in text.

The ten most frequently occurring bigrams are displayed in the dashboard.

---

## 📈 Engagement Analysis

Reddit is a place where people are allowed to write lengthy thoughts and opinions so I was interested to explore whether the amount someone wrote was associated with Reddit engagement; if people liked longer commments more regardless of their sentiment. 

Comment length was then plotted against the number of upvotes received. 

This visualization allows unusually popular comments to be identified while also providing a view of the broader relationship between comment length and engagement.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** — data manipulation and analysis
* **NumPy** — numerical operations
* **NLTK** — tokenization, POS tagging, lemmatization, and NLP processing
* **VADER** — sentiment analysis
* **Matplotlib** — data visualization
* **WordCloud** — word-cloud generation
* **Pillow (PIL)** — image-mask processing
* **Streamlit** — interactive dashboard development
* **PRAW** — Reddit API access

---

## 📁 Project Structure

```text
Barbie-Reddit-Sentiment-Analysis/
│
├── app.py
├── requirements.txt
├── barbie_cleaned2.csv
├── barbie_final_mask.png
└── README.md
```

---

## 🚀 Running the Project Locally

Clone the repository:

```bash
git clone YOUR-GITHUB-REPOSITORY-URL
```

Move into the project directory:

```bash
cd YOUR-REPOSITORY-NAME
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

The dashboard should then open in your browser.

---

## ⚠️ Limitations

This project has several limitations that should be considered when interpreting the results.

**VADER classification:** Sentiment analysis is an approximation. Reddit comments can contain sarcasm, humor, slang, cultural references, and context that rule-based sentiment models may not interpret correctly.

**Engagement is not sentiment:** Upvotes were incorporated into the selection of strongly positive comments to identify reactions that resonated with the community. However, an upvote does not necessarily mean that a user agrees with a comment's sentiment. Since Reddit does not allow downloading downvotes, reliance on upvotes for user engagment should be considered with caution.

**Reddit is not representative of the general population:** The analysis represents users participating in the selected Reddit discussions and should not be interpreted as representative of all viewers of *Barbie*.


---

## 💡 What I Learned

This project provided hands-on experience with the complete analytics workflow, including:

* Collecting data through an API
* Cleaning and preprocessing unstructured text
* Applying NLP techniques
* Evaluating the limitations of automated sentiment classification
* Developing custom analytical criteria based on exploratory findings
* Building an interactive Streamlit application
* Deploying a Python analytics project as a live web application



---

## 📚 Sources

Add the sources used to guide the methodology and analysis here.

Examples may include:

* VADER sentiment analysis documentation/paper
* NLTK documentation
* PRAW documentation
* Reddit API documentation
* Any articles or research papers used to support methodological decisions

---

## 👩‍💻 Author

**Shifa Maqsood**

Data analytics project exploring NLP, audience sentiment, and online community engagement.
