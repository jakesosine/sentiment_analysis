"""
Note to self: To add the python interpretor path to the standalone spyder IDE do the following
1. import sys
2. run print(sys.executable)
3. Navigate to Spyder -> Preferences -> Python Interpretor 
4. Copy result of sys.executable
5. Paste path in "Use the following python interpretor"
"""
import pandas as pd 
import numpy as np
#import nltk
#nltk.download('vader_lexicon') #downloading vader lexicon
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time



vader = SentimentIntensityAnalyzer()

sample1 = 'I really love working here'
sample2 = 'I really hate working here'


lst = [sample1,sample2]


for i in lst:
    scores = vader.polarity_scores(i)
    print(scores['compound'])
    
