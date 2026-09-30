#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis in Python
# # narges haghani
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
import nltk


# In[2]:


import sys
get_ipython().system('{sys.executable} -m pip install matplotlib seaborn nltk')


# In[3]:


import sys
print(sys.executable)


# In[4]:


import sys
get_ipython().system('{sys.executable} -m pip install pandas')


# In[5]:


import pandas as pd
print(pd.__version__)


# In[6]:


# Read in data
df = pd.read_csv(r'C:\Users\UTOB\Desktop\uni project\reviews\reviews.csv')
print(df.shape)
df = df.head(500)
print(df.shape)


# In[7]:


df.head()


# In[8]:


ax = df['Score'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()


# In[9]:


example = df['Text'][50]
print(example)


# In[10]:


import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_percrptron_tagger')
nltk.download('stopwords')
nltk.download('vader_lexicam')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')


# In[11]:


import nltk
nltk.download('punkt', download_dir=r'C:\Users\UTOB\AppData\Roaming\nltk_data')


# In[12]:


nltk.data.path.append(r'C:\Users\UTOB\AppData\Roaming\nltk_data')


# In[13]:


from nltk.tokenize import word_tokenize

example = "This is an example sentence for tokenization."
tokens = word_tokenize(example)
print(tokens[:10])


# In[14]:


tokens = nltk.word_tokenize(example)
tokens[:10]


# In[15]:


tagged = nltk.pos_tag(tokens)
tagged[:10]


# In[16]:


entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()


# In[17]:


import nltk
nltk.download('vader_lexicon')


# In[18]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()


# In[19]:


sia.polarity_scores('I am so happy!')


# In[20]:


sia.polarity_scores('This is the worst thing ever.')


# In[21]:


sia.polarity_scores(example)


# In[22]:


get_ipython().system('pip install ipywidgets')


# In[23]:


get_ipython().system('jupyter nbextension enable --py widgetsnbextension')


# In[24]:


get_ipython().system('jupyter labextension list')


# In[25]:


get_ipython().system('pip install ipywidgets==7.7.1')
get_ipython().system('jupyter nbextension enable --py widgetsnbextension')


# In[26]:


get_ipython().run_line_magic('pip', 'install --upgrade jupyter ipywidgets widgetsnbextension')


# In[27]:


# Run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)


# In[28]:


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')


# In[29]:


# Now we have sentiment score and metadata
vaders.head()


# In[30]:


ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compund Score by Amazon Star Review')
plt.show()


# In[31]:


fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


# In[32]:


get_ipython().run_line_magic('pip', 'install transformers')
get_ipython().run_line_magic('pip', 'install scipy')



# In[33]:


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


# In[34]:


get_ipython().system('pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')


# In[35]:


get_ipython().run_line_magic('pip', 'install ipywidgets tqdm')
get_ipython().run_line_magic('pip', 'install torch torchvision torchaudio')


# In[36]:


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[37]:


# VADER results on example
print(example)
sia.polarity_scores(example)


# In[38]:


# Run for Roberta Model
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)


# In[39]:


def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict


# In[40]:


res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')


# In[41]:


results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')


# In[42]:


results_df.columns


# In[43]:


sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10')
plt.show()


# In[44]:


results_df.query('Score == 1') \
    .sort_values('roberta_pos', ascending=False)['Text'].values[0]


# In[45]:


results_df.query('Score == 1') \
    .sort_values('vader_pos', ascending=False)['Text'].values[0]


# In[46]:


# nevative sentiment 5-Star view


# In[47]:


results_df.query('Score == 5') \
    .sort_values('roberta_neg', ascending=False)['Text'].values[0]


# In[48]:


results_df.query('Score == 5') \
    .sort_values('vader_neg', ascending=False)['Text'].values[0]


# In[49]:


# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")


# In[50]:


# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")


# In[51]:


from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")


# In[52]:


sent_pipeline('I love sentiment analysis!')


# In[53]:


sent_pipeline('Make sure to like and subscribe!')


# In[54]:


sent_pipeline('booo')


# In[55]:


sent_pipeline('it is hard and veryyyyyy bad')


# In[56]:


sent_pipeline('its so good')


# In[57]:


sent_pipeline('its a great thing')

input("Press Enter to exit...")

