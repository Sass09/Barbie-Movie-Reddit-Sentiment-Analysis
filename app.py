import ast
import collections

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from nltk import bigrams
from PIL import Image
from wordcloud import WordCloud


# --------------------------------------------------
# Page setup
# --------------------------------------------------

st.set_page_config(
    
    page_title="Barbie Reddit Sentiment Analysis"
    ,layout="wide"
)
st.markdown("<h1 style='text-align: center;'>🎬 Barbie Movie Reddit Sentiment Analysis</h1>", unsafe_allow_html=True)

#st.title("🎬 Barbie Movie Reddit Sentiment Analysis")

st.markdown(
    """
This Barbie Movie Reddit Post Analysis showcases the consumer sentiment by using VADER and Text Blob for the  sentiment analysis. 
    """
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv("barbie_cleaned2.csv")


# --------------------------------------------------
# Summary metrics
# --------------------------------------------------

left, col1, col2, col3, col4, right = st.columns(
    [1, 2, 2, 2, 2, 1]
)

with col1:
    st.metric(
        "Median Upvotes",
        f"{df['Upvotes'].median():.1f}"
    )

with col2:
    st.metric(
        "Average Upvotes",
        f"{df['Upvotes'].mean():.1f}"
    )

with col3:
    average_positive = (
        df.loc[
            df["Sentiment_Type"] == "Positive",
            "compound"
        ]
        .mean()
    )

    st.metric(
        "Average Positive Sentiment",
        f"{average_positive:.2f}"
    )

with col4:
    average_negative = (
        df.loc[
            df["Sentiment_Type"] == "Negative",
            "compound"
        ]
        .mean()
    )

    st.metric(
        "Average Negative Sentiment",
        f"{average_negative:.2f}"
    )


# --------------------------------------------------
# Prepare tokens and bigrams
# --------------------------------------------------

flattened_list = []

for token_string in df["cleaned_tokens"].dropna():
    tokens = ast.literal_eval(token_string)
    flattened_list.extend(tokens)

count_words = collections.Counter(flattened_list)

bigram_list = list(bigrams(flattened_list))
bigram_count = collections.Counter(bigram_list)

text = " ".join(flattened_list)


# --------------------------------------------------
# Chart functions
# --------------------------------------------------

def word_cloud(text):
    mask = np.array(
        Image.open("barbie_final_mask.png").convert("L")
    )

    mask = np.where(mask == 255, 0, 255).astype(np.uint8)

    wc = WordCloud(
        background_color="white",
        mask=mask,
        contour_color="black",
        contour_width=2,
        colormap="RdPu",
    ).generate(text)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(
        wc,
        interpolation="bilinear",
        aspect="auto"
    )

    ax.axis("off")
    fig.tight_layout(pad=0)

    return fig


def sentiment_pie_chart(df):
    sentiment_counts = df["Sentiment_Type"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 8))

    wedges, texts, autotexts = ax.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct="%2.1f%%",
        startangle=90,
        colors=[
            "#E0218A",
            "#F8D7E8",
            "#87CEEB"
        ],
        wedgeprops={
            "width": 0.30,
            "edgecolor": "white"
        },
        labeldistance=0.78,
        pctdistance=0.58
    )

    # Make labels bigger
    for text in texts:
        text.set_fontsize(14)

    # Make percentages bigger
    for autotext in autotexts:
        autotext.set_fontsize(13)
        autotext.set_fontweight("bold")

    ax.axis("equal")
    fig.tight_layout()

    return fig

def comment_len(df):
    chart_df = df.copy()

    chart_df["Comment_Length"] = (
        chart_df["LowercaseComments"]
        .fillna("")
        .str.len()
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        chart_df["Comment_Length"],
        chart_df["Upvotes"],
        alpha=0.6,
        s=40,
        color="#E0218A"
    )

    ax.set_xlabel("Comment Length (characters)")
    ax.set_ylabel("Upvotes")

    ax.grid(
        True,
        alpha=0.3
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return fig


def top_bigrams_chart(bigram_count):
    top_10 = bigram_count.most_common(10)

    labels = []
    counts = []

    for bigram, count in top_10:
        labels.append(" ".join(bigram))
        counts.append(count)

    fig, ax = plt.subplots(figsize=(8, 8))

    bars = ax.barh(
        labels,
        counts,
        color="#E0218A",
        edgecolor="#C2185B"
    )

    ax.bar_label(
        bars,
        padding=3
    )

    ax.set_xlabel("Frequency")
    ax.invert_yaxis()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.grid(
        axis="x",
        alpha=0.2
    )

    fig.tight_layout()

    return fig


# --------------------------------------------------
# Positive and negative comment functions
# --------------------------------------------------

def isactually_positive(row, threshold):
    return (
        row["compound"] > 0.7
        and row["Upvotes"] >= threshold
    )


def get_top_positive(df):
    threshold = df["Upvotes"].quantile(0.8)

    filtered_df = df.copy()

    filtered_df["is_positive"] = filtered_df.apply(
        lambda row: isactually_positive(
            row,
            threshold
        ),
        axis=1
    )

    top_positive = (
        filtered_df[filtered_df["is_positive"]]
        .sort_values(
            by="Upvotes",
            ascending=False
        )
        .head(10)
    )

    return top_positive


def isactually_negative(row):
    score = row["scores"]

    if isinstance(score, str):
        score = ast.literal_eval(score)

    return (
        score["compound"] < -0.8
        and score["neu"] >= 0.7
    )


def get_top_negative(df):
    filtered_df = df.copy()

    filtered_df["is_negative"] = filtered_df.apply(
        isactually_negative,
        axis=1
    )

    top_negative = (
        filtered_df[filtered_df["is_negative"]]
        .sort_values(
            by="Upvotes",
            ascending=False
        )
        .head(10)
    )

    return top_negative


# --------------------------------------------------
# Dashboard chart grid
# --------------------------------------------------

left, chart_col1, chart_col2, right = st.columns(
    [1, 4, 4, 1],
    gap="large"
)

with chart_col1:
    st.subheader("Barbie Word Cloud")
    st.caption(
        "Most frequently used words in the Reddit discussion."
    )

    st.pyplot(
        word_cloud(text),
        width="stretch"
    )

with chart_col2:
    st.subheader("Overall Sentiment")
    st.caption(
        "Distribution of Reddit comments using VADER sentiment analysis."
    )

    st.pyplot(
        sentiment_pie_chart(df),
        width="stretch"
    )


left, chart_col3, chart_col4, right = st.columns(
    [1, 4, 4, 1],
    gap="large"
)

with chart_col3:
    st.subheader("Top Discussion Themes")
    st.caption(
        "Most common two-word phrases in the Reddit discussion."
    )

    st.pyplot(
        top_bigrams_chart(bigram_count),
        width="stretch"
    )

with chart_col4:
    st.subheader("Comment Length vs. Upvotes")
    st.caption(
        "Relationship between comment length and number of upvotes."
    )

    st.pyplot(
        comment_len(df),
        width="stretch"
    )


# --------------------------------------------------
# Positive comments table
# --------------------------------------------------

top_positive = get_top_positive(df)

display_positive = (
    top_positive[
        [
            "LowercaseComments",
            "Upvotes",
            "compound"
        ]
    ]
    .rename(
        columns={
            "LowercaseComments": "Comment",
            "compound": "Sentiment Score"
        }
    )
)

st.subheader(
    "Most Upvoted Strongly Positive Comments"
)

st.dataframe(
    display_positive,
    hide_index=True,
    width="stretch"
)


# --------------------------------------------------
# Negative comments table
# --------------------------------------------------

top_negative = get_top_negative(df)

display_negative = (
    top_negative[
        [
            "LowercaseComments",
            "Upvotes",
            "compound"
        ]
    ]
    .rename(
        columns={
            "LowercaseComments": "Comment",
            "compound": "Sentiment Score"
        }
    )
)

st.subheader(
    "Most Upvoted Strongly Negative Comments"
)

st.dataframe(
    display_negative,
    hide_index=True,
    width="stretch"
)