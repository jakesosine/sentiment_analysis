"""
Note to self: To add the python interpretor path to the standalone spyder IDE do the following
1. import sys
2. run print(sys.executable)
3. Navigate to Spyder -> Preferences -> Python Interpretor 
4. Copy result of sys.executable
5. Paste path in "Use the following python interpretor"
"""

#import required packages and download vader lexicon if needed
import pandas as pd 
import numpy as np
#import nltk
#nltk.download('vader_lexicon') 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns 
#creating our vader object
vader = SentimentIntensityAnalyzer()

# setting up some sample sentences that we can start to look at. 
sample1 = 'I really love working here'
sample2 = 'I really hate working here'
sample3 = 'I have never hated a company more than I have hated this hellhole'
sample4 = 'I have never felt more appreciated than when I come to work each day'


lst = [sample1,sample2,sample3, sample4]


for i in lst: # iterate through the list 
    scores = vader.polarity_scores(i) # gather the scores 
    print(scores)# print the dictionary
    print(scores['compound'])#print the compound
    
# read in the data of a wine review list
df = pd.read_csv('../data/winemag-data_first150k.csv')
# Inspect the data
print(df.head())
# Describe the dataframe 
print(df.describe())
# Apply the vader polarity scores and make a new dataframe column 
df['vader'] =df['description'].apply(lambda x: vader.polarity_scores(x))
# make a compound  column in the dataframe
df['compound'] = df['vader'].apply(lambda x: x['compound'])
# apply sentiment label to datarame column 

def sentiment_label(compound):
    if compound >=.5:
        return 'Positive'
    elif compound <.5:
        return 'Negative'
df['positive_or_negative'] = df['compound'].apply(lambda x: sentiment_label(x))
var_3 = df['variety'].isin(['Chardonnay','Pinot Noir','Cabernet Sauvignon'])
df = df[var_3]

g = sns.catplot(x="variety", y="points", hue="positive_or_negative",
               data=df, kind="violin")

plt.show()
g.savefig('../figures/fig_1.png')
sns.boxplot(x="variety", y="points", hue="positive_or_negative",
               data=df)

