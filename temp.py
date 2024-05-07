import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from gensim import corpora, models
from itertools import combinations
import pickle

with open("concepts", "rb") as file:
    phrases  = pickle.load(file)


# Preprocessing: tokenize phrases
tokenized_phrases = [phrase.split() for phrase in phrases]

# Create dictionary and corpus
dictionary = corpora.Dictionary(tokenized_phrases)
corpus = [dictionary.doc2bow(phrase) for phrase in tokenized_phrases]

# Run LDA
lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

# Preprocessing: tokenize phrases
tokenized_phrases = [phrase.split() for phrase in phrases]

# Create dictionary and corpus
dictionary = corpora.Dictionary(tokenized_phrases)
# corpus = [dictionary.doc2bow(phrase) for phrase in tokenized_phrases]

# Run LDA
lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

import matplotlib.pyplot as plt

# # Get the top-N most important topics
# top_topics = lda_model.show_topics(num_topics=10, formatted=False)

# topic_words = [', '.join([word[0] for word in topic[1]]) for topic in top_topics]
# print(topic_words)

# # Extract topic indices and weights
# topic_indices = [topic[0] for topic in top_topics]
# topic_weights = [sum([item[1] for item in topic[1]]) for topic in top_topics]

# # Plot the most important topics
# plt.figure(figsize=(10, 6))
# plt.bar(topic_indices, topic_weights, color='skyblue')
# plt.xlabel('Topic')
# plt.ylabel('Weight')
# plt.title('Topic Importance Weights for Each Topic')
# plt.xticks(topic_indices)
# plt.savefig("topics.png")
# plt.show()


from wordcloud import WordCloud

# Choose a topic index
topic_index = 2  # Adjust this according to the topic you want to visualize

# Get the word probabilities for the chosen topic
topic_words_probs = lda_model.show_topic(topic_index, topn=50)

# Convert the word probabilities to a dictionary
wordcloud_dict = {word: prob for word, prob in topic_words_probs}

# Create a word cloud object
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_dict)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title(f'Most Important Words (Shown By Size) In Topic {topic_index}')
plt.axis('off')
plt.savefig("topic2.png")
plt.show()

