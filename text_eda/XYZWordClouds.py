import sys
import numpy as np
from PIL import Image
from os import path
import matplotlib.pyplot as plt
import os
import random
import csv
import pandas as pd

from wordcloud import WordCloud

import glob

if __name__=='__main__':
    for inputCSV in glob.glob('NeighborhoodOutputs/*.csv'):
        filename = os.path.basename(inputCSV)
        x = os.path.splitext(filename)[0]
        y = (x.split('.')[0])
        text = pd.read_csv(inputCSV)
        data = dict(zip(text['word'].tolist(), text['count'].tolist()))
        
        Neighborhood = np.array(Image.open('NeighborhoodPics/' + y + ".png"))
        Neighborhood = Neighborhood*255
        
        wc = WordCloud(background_color="white", max_words=1000, mask=Neighborhood, contour_width=3, contour_color='firebrick').generate_from_frequencies(data)
        wc.to_file('wordclouds/' + y + '.png')
        
        #df100.to_csv('../NeighborhoodOutputs/'+inputCSV + '_output.csv', index=False)
