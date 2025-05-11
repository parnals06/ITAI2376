#!/usr/bin/env python
# coding: utf-8

# <center><img src="images/logo.png" alt="drawing" width="400" style="background-color:white; padding:1em;" /></center> <br/>
# 
# # Application of Deep Learning to Text and Image Data
# ## Module 2, Lab 5: Finetuning BERT
# 
# 
# BERT stands for **B**idirectional **E**ncoder **R**epresentations from **T**ransformers. To learn how BERT works, let's fine-tune the __BERT__ model to classify product reviews. You will use a new library called __transformers__ to download a pre-trained BERT model. 
# 
# You will learn:
# 
# - How to load and format the dataset
# - How to load the pre-trained model
# - How to train and test the model
# 
# __BERT and its variants use more resources than the other models you have used so far. This may cause your instance to run out of memory. If that happens:__
# 
# - Restart the kernel (Kernel->Restart from the top menu)
# - Reduce the batch size 
# - Then re-run the code
# 
# 
# 
# __Note__: In this walkthrough, you will use a light version of the original BERT implementation called __"DistilBert"__. You can checkout [the paper](https://arxiv.org/pdf/1910.01108.pdf) about it for more details. 
# 
# ---
# This lab uses a dataset derived from a small sample of Amazon product reviews. 
# 
# __Review dataset schema:__
# * __reviewText:__ Text of the review
# * __summary:__ Summary of the review
# * __verified:__ Whether the purchase was verified (True or False)
# * __time:__ UNIX timestamp for the review
# * __log\_votes:__ Logarithm-adjusted votes log(1+votes)
# * __isPositive:__ Whether the review is positive or negative (1 or 0)
# 
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
# 1. [Reading and formatting the dataset](#Reading-and-formatting-the-dataset)
# 1. [Loading the pre-trained model](#Loading-the-pre-trained-model)
# 1. [Training and testing the model](#Training-and-testing-the-model)
# 1. [Getting predictions on the test data](#Getting-predictions-on-the-test-data)

# In[1]:


get_ipython().run_cell_magic('capture', '', '\n!pip install -U -q -r requirements.txt\n')


# In[2]:


get_ipython().system('pip install transformers==4.39.3')


# ## Reading and formatting the dataset
# 
# First, you need to read in the product review dataset and prepare it for the BERT model. To keep the training time down, you will only use the first 2000 data points from the dataset. If you want to improve your model after you understand how to train, you can use more data to train a new model.

# In[3]:


import os
import sys
import time
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from torch.utils.data import DataLoader

# Import system library and append path
sys.path.insert(1, '..')

# Setting tokenizer parallelism to false
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import utility functions that provide answers to challenges
from MLUDTI_EN_M2_Lab5_quiz_questions import *


# Read the dataset.

# In[4]:


df = pd.read_csv("data/NLP-REVIEW-DATA-CLASSIFICATION-TRAINING.csv")


# Print the dataset information to see the field types.

# In[5]:


df.info()


# You do not need any of the rows that do not have __reviewText__, so drop them.

# In[6]:


df.dropna(subset=["reviewText"], inplace=True)


# <div style="border: 4px solid coral; text-align: center; margin: auto;">
#     <h2><i>Try it Yourself!</i></h2>
#     <br>
#     <p style="text-align:center;margin:auto;"><img src="images/activity.png" alt="Activity" width="100" /> </p>
#     <p style=" text-align: center; margin: auto;">Answer the question below to test your understanding of epochs and learning rate.</p>
#     <br>
# </div>

# In[7]:


# question_1


# BERT requires a lot of compute power for large datasets. To reduce the amount of time it takes to train the model, you will only use the first 2,000 data points for this lab. 

# In[8]:


df = df.head(2000)


# Now split the dataset into training and validation data sets, keeping 10% of the data for validation.

# In[9]:


# This separates 10% of the entire dataset into validation dataset.
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["reviewText"].tolist(),
    df["isPositive"].tolist(),
    test_size=0.10,
    shuffle=True,
    random_state=324,
    stratify = df["isPositive"].tolist(),
)


# You need to tokenize the data. To do this, use a special tokenizer built for the DistilBERT model to tokenize the training and validation texts. 

# In[10]:


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts,
                            truncation=True,
                            padding=True)
val_encodings = tokenizer(val_texts,
                          truncation=True,
                          padding=True)


# Create a new `ReviewDataset` class to use with the BERT model. Later, you use the training and validation encoding-label pairs with this new class.

# In[11]:


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx]).to(device)
        return item

    def __len__(self):
        return len(self.labels)
    
train_dataset = ReviewDataset(train_encodings, train_labels)
val_dataset = ReviewDataset(val_encodings, val_labels)


# ## Loading the pre-trained model
# 
# Now, you need to load the model. When you do this, several warnings will print that are related to the last classification layer of BERT where you are using a randomly initialized layer. You can ignore the warnings as they are not relevant to the type of training you are doing.

# In[12]:


model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                            num_labels=2)


# The last step is to freeze all weights until the very last classification layer in the BERT model. This helps accelerate the training process. Training the weights of the whole network (66 million weights) takes a long time. Additionally, 2000 data points would not be enough for that task. Instead, the code below freezes all the weights until the last classification layer. This means only a small portion of the weights gets updated (rest stays the same). This is a common practice with large language models.

# In[13]:


# Freeze the encoder weights until the classifier
for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False


# ## Training and testing the model
# 
# Now that your data is ready and you have configured your model, its time to start the fine-tuning process. This code will take __a long time__ (30+ minutes) to complete with large datasets, that is why you are running it on a subset of the full review dataset.

# First, define the accuracy function.

# In[14]:


def calculate_accuracy(output, label):
    """Calculate the accuracy of the trained network. 
    output: (batch_size, num_output) float32 tensor
    label: (batch_size, ) int32 tensor """
    
    return (output.argmax(axis=1) == label.float()).float().mean()


# Now you need to create the tranining and validation loop. This loop will be similar to the previous train/validation loops, however there are a few extra parameters needed due to the transformer architecture. 
# 
# You need to use the `attention_mask` and get the loss from the output of the model with `loss = output[0]`

# In[15]:


# Hyperparameters
num_epochs = 10
learning_rate = 0.005

# Get the compute device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create data loaders
train_loader = DataLoader(train_dataset, shuffle=True,
                          batch_size=16, drop_last=True)
validation_loader = DataLoader(val_dataset, batch_size=8,
                               drop_last=True)

# Setup the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

model = model.to(device)

for epoch in range(num_epochs):
    
    train_loss, val_loss, train_acc, valid_acc = 0., 0., 0., 0.
    
    start = time.time()
    # Training loop starts
    model.train() # put the model in training mode
    for batch in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Put data, label and attention mask to the correct device
        data = batch["input_ids"].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch["labels"].to(device)
        
        # Make forward pass
        output = model(data, attention_mask=attention_mask, labels=label)
        
        # Calculate the loss (this comes from the output)
        loss = output[0]
        # Make backwards pass (calculate gradients)
        loss.backward()
        # Accumulate training accuracy and loss
        train_acc += calculate_accuracy(output.logits, label).item()
        train_loss += loss.item()
        # Update weights
        optimizer.step()
    
    # Validation loop:
    # This loop tests the trained network on validation dataset
    # No weight updates here
    # torch.no_grad() reduces memory usage when not training the network
    model.eval() # Activate evaluation mode
    with torch.no_grad():
        for batch in validation_loader:
            data = batch["input_ids"].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch["labels"].to(device)
            # Make forward pass with the trained model so far
            output = model(data, attention_mask=attention_mask, labels=label)
            # Accumulate validation accuracy and loss
            valid_acc += calculate_accuracy(output.logits, label).item()
            val_loss += output[0].item()
        
    # Take averages
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    val_loss /= len(validation_loader)
    valid_acc /= len(validation_loader)
    
    end = time.time()
    
    print("Epoch %d: train loss %.3f, train acc %.3f, val loss %.3f, val acc %.3f, seconds % .3f " % (
        epoch+1, train_loss, train_acc, val_loss, valid_acc, end-start))


# ### Looking at what's going on
# 
# The fine-tuned BERT model is able to correctly classify the sentiment of the most of the records in the validation set. Let's observe in more detail how the sentences are tokenized and encoded. You can do this by picking one sentence as example to look at.

# In[16]:


st = val_texts[19]
print(f"Sentence: {st}")
tok = tokenizer(st, truncation=True, padding=True)
print(f"Encoded Sentence: {tok['input_ids']}")


# Print the vocabulary size.

# In[17]:


# The mapped vocabulary is stored in tokenizer.vocab
tokenizer.vocab_size


# Use the encoded sentence with the tokenizer to recover the original sentence. 

# In[18]:


# Methods convert_ids_to_tokens and convert_tokens_to_ids allow to see how sentences are tokenized
print(tokenizer.convert_ids_to_tokens(tok["input_ids"]))


# ## Getting predictions on the test data
# 
# After the model is trained, you can focus on getting test data to make predictions with. Do this by:
# - Reading and format the test dataset
# - Passing the test data to your trained model and make predictions

# In[19]:


# Read the test data (It doesn't have the isPositive label)
df_test = pd.read_csv("data/NLP-REVIEW-DATA-CLASSIFICATION-TEST.csv")
df_test.head()


# Just as before, drop the rows that don't have the __reviewText__.

# In[20]:


df_test.dropna(subset=["reviewText"], inplace=True)


# Making predictions will also take a long time with this model. To get results quickly, start by only making predictions with 15 datapoints from the test set.

# In[21]:


test_texts = df_test["reviewText"].tolist()[0:15]


# In[22]:


test_encodings = tokenizer(test_texts,
                           truncation=True,
                           padding=True)


# Create labels for the test dataset to pass zeros using `[0]*len(test_texts)`.

# In[23]:


test_dataset = ReviewDataset(test_encodings, [0]*len(test_texts))


# Then, create a dataloader for the test set and record the corresponding predictions.

# In[24]:


test_loader = DataLoader(test_dataset, batch_size=4)
test_predictions = []
model.eval()

with torch.no_grad():
    for batch in test_loader:
        data = batch["input_ids"].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch["labels"].to(device)
        output = model(data, attention_mask=attention_mask, labels=label)
        predictions = torch.argmax(output.logits, dim=-1)
        test_predictions.extend(predictions.cpu().numpy())


# Finally, pick an example sentence and examine the prediction. Does the prediction look correct? 
# 
# Remember 
# 
# - 1->positive class 
# - 0->negative class

# In[25]:


k = 13
print(f'Text: {test_texts[k]}')
print(f'Prediction: {test_predictions[k]}')


# <div style="border: 4px solid coral; text-align: center; margin: auto;">
#     <h2><i>Try it Yourself!</i></h2>
#     <br>
#     <p style="text-align:center; margin:auto;"><img src="images/challenge.png" alt="Challenge" width="100" /> </p>
#     <p style="margin: auto; text-align: center; margin: auto;">You trained the model for 10 epochs. Would you get better results from the validation dataset if the model trained longer?</p> <br>
#     <p style="margin: auto; text-align: center; margin: auto;">Make a note of your last <code> Val_loss </code> result.</p> 
#     <p style="margin: auto; text-align: center; margin: auto;">Then, in the <a href="#Training-and-testing-the-model">Training and testing the model</a> section, change the <code> num_epochs </code> parameter to <code>20</code>.</p>
#     <p style="margin: auto; text-align: center; margin: auto;">Finally, re-run the code blocks to load the pre-trained model, and train your model.</p>
#     <p style="margin: auto; text-align: center; margin: auto;">Did <code>Val_loss</code> improve?</p>
#     </ol>    
# </div>

# In[26]:


# Hyperparameters
num_epochs = 20 # change number of epochs to 20
learning_rate = 0.005

# Get the compute device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create data loaders
train_loader = DataLoader(train_dataset, shuffle=True,
                          batch_size=16, drop_last=True)
validation_loader = DataLoader(val_dataset, batch_size=8,
                               drop_last=True)

# Setup the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

model = model.to(device)

for epoch in range(num_epochs):
    
    train_loss, val_loss, train_acc, valid_acc = 0., 0., 0., 0.
    
    start = time.time()
    # Training loop starts
    model.train() # put the model in training mode
    for batch in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Put data, label and attention mask to the correct device
        data = batch["input_ids"].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch["labels"].to(device)
        
        # Make forward pass
        output = model(data, attention_mask=attention_mask, labels=label)
        
        # Calculate the loss (this comes from the output)
        loss = output[0]
        # Make backwards pass (calculate gradients)
        loss.backward()
        # Accumulate training accuracy and loss
        train_acc += calculate_accuracy(output.logits, label).item()
        train_loss += loss.item()
        # Update weights
        optimizer.step()
    
    # Validation loop:
    # This loop tests the trained network on validation dataset
    # No weight updates here
    # torch.no_grad() reduces memory usage when not training the network
    model.eval() # Activate evaluation mode
    with torch.no_grad():
        for batch in validation_loader:
            data = batch["input_ids"].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch["labels"].to(device)
            # Make forward pass with the trained model so far
            output = model(data, attention_mask=attention_mask, labels=label)
            # Accumulate validation accuracy and loss
            valid_acc += calculate_accuracy(output.logits, label).item()
            val_loss += output[0].item()
        
    # Take averages
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    val_loss /= len(validation_loader)
    valid_acc /= len(validation_loader)
    
    end = time.time()
    
    print("Epoch %d: train loss %.3f, train acc %.3f, val loss %.3f, val acc %.3f, seconds % .3f " % (
        epoch+1, train_loss, train_acc, val_loss, valid_acc, end-start))


# ### Looking at what's going on
# 
# The fine-tuned BERT model is able to correctly classify the sentiment of the most of the records in the validation set. Let's observe in more detail how the sentences are tokenized and encoded. You can do this by picking one sentence as example to look at.

# In[27]:


st = val_texts[19]
print(f"Sentence: {st}")
tok = tokenizer(st, truncation=True, padding=True)
print(f"Encoded Sentence: {tok['input_ids']}")


# Print the vocabulary size.

# In[28]:


# The mapped vocabulary is stored in tokenizer.vocab
tokenizer.vocab_size


# Use the encoded sentence with the tokenizer to recover the original sentence. 

# In[29]:


# Methods convert_ids_to_tokens and convert_tokens_to_ids allow to see how sentences are tokenized
print(tokenizer.convert_ids_to_tokens(tok["input_ids"]))


# ## Getting predictions on the test data
# 
# After the model is trained, you can focus on getting test data to make predictions with. Do this by:
# - Reading and format the test dataset
# - Passing the test data to your trained model and make predictions

# In[30]:


# Read the test data (It doesn't have the isPositive label)
df_test = pd.read_csv("data/NLP-REVIEW-DATA-CLASSIFICATION-TEST.csv")
df_test.head()


# Just as before, drop the rows that don't have the __reviewText__.

# In[31]:


df_test.dropna(subset=["reviewText"], inplace=True)


# Making predictions will also take a long time with this model. To get results quickly, start by only making predictions with 15 datapoints from the test set.

# In[32]:


test_texts = df_test["reviewText"].tolist()[0:15]


# In[33]:


test_encodings = tokenizer(test_texts,
                           truncation=True,
                           padding=True)


# Create labels for the test dataset to pass zeros using `[0]*len(test_texts)`.

# In[34]:


test_dataset = ReviewDataset(test_encodings, [0]*len(test_texts))


# Then, create a dataloader for the test set and record the corresponding predictions.

# In[35]:


test_loader = DataLoader(test_dataset, batch_size=4)
test_predictions = []
model.eval()

with torch.no_grad():
    for batch in test_loader:
        data = batch["input_ids"].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch["labels"].to(device)
        output = model(data, attention_mask=attention_mask, labels=label)
        predictions = torch.argmax(output.logits, dim=-1)
        test_predictions.extend(predictions.cpu().numpy())


# Finally, pick an example sentence and examine the prediction. Does the prediction look correct? 
# 
# Remember 
# 
# - 1->positive class 
# - 0->negative class

# In[36]:


k = 13
print(f'Text: {test_texts[k]}')
print(f'Prediction: {test_predictions[k]}')


# ----
# ## Conclusion
# 
# In this lab you learned how to import a pre-trained Transformer model and fine-tune it for a specific task. Although you used a lighter version of the BERT model, these types of models tend to use large amounts of compute power. For that reason, you only worked with the first 2000 datapoints of the dataset. To see more general results, you need to spend more time training while using the whole dataset. 
# 
# ## Next Lab: Reading and plotting images
# In the next lab you will learn how to read images and plot them as you start to learn about computer vision problems.
# 
