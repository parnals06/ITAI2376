#!/usr/bin/env python
# coding: utf-8

# <center><img src="images/logo.png" alt="drawing" width="400" style="background-color:white; padding:1em;" /></center> <br/>
# 
# # Application of Deep Learning to Text and Images
# ## Module 2, Lab 3: GloVe Word Vectors
# 
# This notebook supports the topics presented on on the __Word Embeddings__ lecture.
# 
# In this lab you will learn how to use word embeddings. Word embeddings, or word vectors, are a way of representing words as numeric vectors in a high-dimensional space. These embeddings capture the meaning of the words, the relationships between them, and can be used as inputs to machine learning models for a variety of natural language processing tasks.
# 
# The term __Word vectors__ refers to a family of related techniques, first gaining popularity via `Word2Vec` which associates an $n$-dimensional vector to every word in the target language.
# 
# - __Note__: Normally $n$ is in the range of $50$ to $500$. In this lab, you will set it to $50$
# 
# 
# You will learn:
# - What GloVe word vectors are
# - How to load GloVe word vectors
# - How to use GloVe to produce word vectors
# - What cosine Similarity is
# - How to use cosine similarity to compare words
# 
# ---
# 
# You will be presented with two kinds of exercises throughout the notebook: activities and challenges. <br/>
# 
# | <img style="float: center;" src="images/activity.png" alt="Activity" width="125"/>| <img style="float: center;" src="images/challenge.png" alt="Challenge" width="125"/>|
# | --- | --- |
# |<p style="text-align:center;">No coding is needed for an activity. You try to understand a concept, <br/>answer questions, or run a code cell.</p> |<p style="text-align:center;">Challenges are where you can practice your coding skills.</p> |
# 

# ## Index
# 
# 1. [GloVe Word Vectors](#GloVe-Word-Vectors)
# 1. [Cosine Similarity](#Cosine-Similarity)

# First, install the latest versions of the libraries.

# In[1]:


# installing libraries
get_ipython().system('pip install -U -q -r requirements.txt')


# In[2]:


from torchtext.vocab import GloVe

#from torchtext.vocab import GloVe
GloVe.url['6B'] = 'https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip'

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ## GloVe Word Vectors
# 
# You learned about __Word2Vec__ and __FastText__ as word embedding techniques. Now you will use a set of pre-trained word embeddings. Pre-trained embeddings are created by someone else who took the time and computational power to train. This reduces your cost by not having to train the model yourself. One popular word embedding is __GloVe__ embeddings. GloVe is a variation of a Word2Vec model. To learn more about GloVe, read the [Project GloVe](https://nlp.stanford.edu/projects/glove/) website.
# 
# In this exercise, you will discover relationships between word vectors using the GloVe embeddings. 

# You can easily import GloVe embeddings from the Torchtext library. Here, you will get vectors with $50$ dimensions. 
# 
# The `name` parameter refers to the particular pre-trained model that should be loaded: 
# 
# - Wikipedia 2014 + Gigaword 5 
#     - 6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download: `"6B"` 
#     - This is the model that you will load.
# - Common Crawl 
#     - 42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download: `"42B"`
# - Common Crawl 
#     - 840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download: `"840B"`
# - Etc 
#     - See documentation in Stanford link above

# <div style="border: 4px solid coral; text-align: center; margin: auto;">
#     <h2><i>Try it Yourself!</i></h2>
#     <br>
#     <p style="text-align:center;margin:auto;"><img src="images/activity.png" alt="Activity" width="100" /> </p>
#     <p style=" text-align: center; margin: auto;">Run the cell below to load the GloVe embedding model and select the dimension.</p>
#     <br>
# </div>

# In[3]:


# Load the model. You can change dim to 50, 100, 300
glove = GloVe(name="6B", dim=50)


# Now that the data is loaded, you can access it and print example word embeddings.

# In[4]:


print(f"cat -> {glove['cat']}\n")


# What do these numbers mean? 
# 
# You might notice that the tensor has 50 values in it. This is related to the dimension flag (`dim=50`) you set when you loaded the GloVe model. You can generate word embeddings for several words and use them to determine how closely related words are. This is a task that machine learning is really good at.

# <div style="border: 4px solid coral; text-align: center; margin: auto;"> 
#     <h2><i>Try it Yourself!</i></h2>
#     <p style="text-align:center; margin:auto;"><img src="images/challenge.png" alt="Challenge" width="100" /> </p>
#     <p style=" text-align: center; margin: auto;">In the code block below, generate word embeddings for the words "computer" and "human" using pre-trained GloVe embedding.</p>
#     <br>
# </div>
# 

# In[5]:


############## CODE HERE ###############

print("computer ->", glove['computer'])
print("human ->", glove['human'])

############## END OF CODE ##################


# ## Cosine Similarity
# 
# You learned about cosine similarity in class, now let's look at an example. Use the [cosine_similarity()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) function from scikit-learn to easily calculate cosine similarity between word vectors.

# <div style="border: 4px solid coral; text-align: center; margin: auto;">
#     <h2><i>Try it Yourself!</i></h2>
#     <br>
#     <p style="text-align:center;margin:auto;"><img src="images/activity.png" alt="Activity" width="100" /> </p>
#     <p style=" text-align: center; margin: auto;">Run the cell below to calculate cosine similarity between word vectors.</p>
#     <br>
# </div>

# In[6]:


# define the similarity between two words
def similarity(w1, w2):
    return cosine_similarity([glove[w1].tolist()], [glove[w2].tolist()])


# Say if w1 is closer to w2 than w3
def simCompare(w1, w2, w3):
    s1 = similarity(w1, w2)
    s2 = similarity(w1, w3)
    if s1 > s2:
        print(f"'{w1}'\tis closer to\t'{w2}'\tthan\t'{w3}'\n")
    else:
        print(f"'{w1}'\tis closer to\t'{w3}'\tthan\t'{w2}'\n")


# In[7]:


simCompare("actor", "pen", "film")
simCompare("cat", "dog", "sea")


# <div style="border: 4px solid coral; text-align: center; margin: auto;"> 
#     <h2><i>Try it Yourself!</i></h2>
#     <p style="text-align:center; margin:auto;"><img src="images/challenge.png" alt="Challenge" width="100" /> </p>
#     <p style=" text-align: center; margin: auto;">Write code to determine if "car" is closer to "truck" than "bike".</p>
#     <br>
# </div>
# 

# In[8]:


############## CODE HERE ###############

simCompare("car", "truck", "bike")

############## END OF CODE ##################


# ----
# ## Conclusion
# 
# You have now seen how to use word embeddings and determine relationships between word vectors using the GloVe embeddings. 
# 
# --- 
# ## Next Lab: Word Embeddings
# In the next lab of this module you will learn how to build a recurrent neural network (RNN) with PyTorch. It will also show you how to implement a simple RNN-based model for natural language processing.
# 
