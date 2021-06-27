import pandas as pd 
import numpy as np
import 

"""
Note: To modify the  python interpretor, import sys and run print(sys.executable) from
the virtual environment. This will require you to change the path in the settings
to the virtual environment exectuable and restart the Kernal
Purpose of the script is to 
"""



#import nltk
#nltk.download('vader_lexicon') #downloading vader lexicon
from nltk.sentiment.vader import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()

sample1 = 'I really love working here'
sample2 = 'I really hate working here'


lst = [sample, sample1,sample2]

vader.polarity_scores(sample)
for i in lst:
    scores = vader.polarity_scores(i)
    print(scores['compound'])
    
