import json
import pandas as pd
import seaborn as sns
import nltk
# from nltk.corpus import stopwords
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gensim import corpora, models
from itertools import combinations
import pickle
# nltk.download('stopwords')

with open('responses.txt', 'r') as file:
    responses = file.read()

calls = responses.split(';')

performances, industries, agendas, key_concepts, labels = tuple(([] for _ in range(5)))


for i in range(len(calls)):
    call = calls[i]
    try:
        folder, content = call.split('*')
    except Exception as e:
        # print(e)
        # print("Didn't work")
        continue
    folder = folder.replace('\n','')
    content = content.replace('```','').replace('json','')
    try:
        data = json.loads(content)
    except:
        # print("Didn't work")
        continue
    performance = data['sentiment_performance']['description']
    industry = data['key_structure']['industry']
    agenda = data['key_structure']['agenda']
    key_concept = data['key_concepts']['topics_discussed']

    # Extract required lists
    path = f"MAEC_Dataset/{folder}"
    df = pd.read_csv("change.csv")
    try:
        label = list(df[df['Folder'] == folder]['Change'])[0]
    except Exception as e:
        continue
    key_concepts += key_concept
    agendas.append(agenda)
    performances.append(performance)
    labels.append(label)

with open('agenda','wb') as file:
    pickle.dump(agendas, file)


# phrases = [
#     "machine learning",
#     "data science",
#     "artificial intelligence",
#     "deep learning",
#     "natural language processing",
#     "neural networks",
#     "computer vision",
#     "big data",
#     "predictive modeling",
#     "pattern recognition"
# ]
# tokenized_phrases = [phrase.split() for phrase in phrases]

# # Create dictionary and corpus
# dictionary = corpora.Dictionary(tokenized_phrases)
# corpus = [dictionary.doc2bow(phrase) for phrase in tokenized_phrases]

# # Run LDA
# lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

# Extract topics
# topics = lda_model.print_topics(num_words=3)
# for topic in topics:
#     print(topic)

# Create concept map
# G = nx.Graph()

# # Add topics as nodes
# for topic_id in range(len(topics)):
#     topic_words = [word.split("*")[1][1:-1] for word in topics[topic_id][1].split(" + ")]
#     G.add_node(f"Topic {topic_id}", label=", ".join(topic_words))

# # Add edges between topics based on co-occurrence of words
# for phrase in tokenized_phrases:
#     for pair in combinations(phrase, 2):
#         for node1 in G.nodes():
#             if pair[0] in G.nodes[node1]["label"]:
#                 for node2 in G.nodes():
#                     if pair[1] in G.nodes[node2]["label"]:
#                         G.add_edge(node1, node2)

# Draw concept map
# pos = nx.spring_layout(G)
# plt.figure(figsize=(10, 8))
# nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=12, font_weight="bold", alpha=0.8)
# plt.title("Concept Map of Topics")
# plt.show()

# Topic-term matrix
# topic_term_matrix = lda_model.get_topics()
# print("Topic-Term Matrix:")
# # print(topic_term_matrix)

# # Document-topic matrix
# document_topic_matrix = np.zeros((len(corpus), lda_model.num_topics))
# for i, doc in enumerate(corpus):
#     topics = lda_model.get_document_topics(doc)
#     for topic, prob in topics:
#         document_topic_matrix[i][topic] = prob
# print("\nDocument-Topic Matrix:")
# # print(document_topic_matrix)

# top_topics = lda_model.show_topics(num_topics=10, formatted=False)

# # Create a list of the top-N topic indices
# top_topic_indices = [topic[0] for topic in top_topics]

# # Topic-term matrix visualization
# top_topic_term_matrix = topic_term_matrix[top_topic_indices]
# plt.figure(figsize=(10, 6))
# sns.heatmap(top_topic_term_matrix, cmap="YlGnBu", xticklabels=dictionary.values(), yticklabels=top_topic_indices)
# plt.title("Top 10 Topic-Term Matrix")
# plt.xlabel("Term")
# plt.ylabel("Topic")
# plt.show()

# # Document-topic matrix visualization
# top_document_topic_matrix = document_topic_matrix[:, top_topic_indices]
# plt.figure(figsize=(10, 6))
# sns.heatmap(top_document_topic_matrix, cmap="YlGnBu", yticklabels=range(len(corpus)), cbar_kws={'label': 'Probability'})
# plt.title("Top 10 Document-Topic Matrix")
# plt.xlabel("Topic")
# plt.ylabel("Document")
# plt.show()


# df3 = pd.DataFrame
# df2 = pd.DataFrame({'X':performances, 'y':labels}).dropna()
# df2.to_csv("data2.csv", index=False)

# # df_flat = pd.DataFrame(df['x'].to_list(), columns=[f'x{i}' for i in range(1, 101)])

# # Concatenate the flattened DataFrame with the label column
# # df_flat['label'] = df['label']

# # Split the dataset into training and testing sets
# X = df_flat.drop('label', axis=1)
# y = df_flat['label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize the logistic regression model
# model = LogisticRegression()

# # Fit the model to the training data
# model.fit(X_train, y_train)

# # Predict on the testing data
# y_pred = model.predict(X_test)

# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# # import pickle

# # with open("f_descriptions", "wb") as file:
# #     pickle.dump(f_descriptions, file)

# # with open("f_labels", 'wb') as file:
# #     pickle.dump(f_labels, file)

# # print(f_descriptions)

# # def remove_stopwords(sentences):
# #     # Get the list of English stopwords
# #     stop_words = set(stopwords.words('english'))

# #     # Remove stop words from each sentence
# #     new_sentences = []
# #     for sentence in sentences:
# #         words = sentence.split()  # Split the sentence into words
# #         filtered_words = [word for word in words if word.lower() not in stop_words]  # Filter out stop words
# #         filtered_sentence = ' '.join(filtered_words)  # Join the filtered words back into a sentence
# #         new_sentences.append(filtered_sentence)
# #     return new_sentences

# # # p_descriptions = remove_stopwords(p_descriptions)




# # import pandas as pd
# # import gensim
# # from sklearn.decomposition import PCA
# # import matplotlib.pyplot as plt


# # Define Word2Vec model
# model = gensim.models.Word2Vec(
#     [p_descriptions, f_descriptions, o_descriptions],
#     vector_size=100,  # Dimension of word vectors
#     window=5,         # Context window size
#     min_count=1,      # Minimum frequency count of words
#     sg=1              # Skip-gram model
# )

# # print(p_descriptions)




# # # Function to create embedding matrix and perform PCA
# def create_embedding_matrix(words):
#     # Create embedding matrix
#     embedding_matrix = [model.wv[word] for word in words if word in model.wv]
#     return embedding_matrix

# # # Function to plot embeddings using PCA
# # def plot_embedding(embedding_matrix, title, labels):
# #     # Perform PCA
# #     pca = PCA(n_components=2)
# #     pca_result = pca.fit_transform(embedding_matrix)

# #     df = pd.DataFrame({
# #         'x': pca_result[:, 0],
# #         'y': pca_result[:, 1],
# #         'label': labels
# #     })
    
# #     # Plot PCA result
# #     plt.figure(figsize=(10, 8))
# #     sns.scatterplot(data=df, x='x', y='y', hue='label', palette='Set2', legend='full')
# #     plt.title(title)
# #     plt.xlabel('Principal Component 1')
# #     plt.ylabel('Principal Component 2')
# #     plt.show()
# # # Create embedding matrices for each list
# # p_embed = create_embedding_matrix(p_descriptions)

# # plot_embedding(p_embed, "Title", p_labels)


# # # f_embed = create_embedding_matrix(f_descriptions)
# # # o_embed = create_embedding_matrix(o_descriptions)
