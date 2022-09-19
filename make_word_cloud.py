import numpy as np
import matplotlib.pyplot as plt
import argparse

from wordcloud import STOPWORDS, WordCloud, ImageColorGenerator
from PIL import Image

def run(input_file):
    with open(input_file) as f:
        text = f.read()

    with open('stop_words.txt', encoding='utf-8') as f: 
        stopwords_list = f.readlines()

    with Image.open('train_image.png') as img:
        train_image = np.array(img)

    image_colors = ImageColorGenerator(train_image)

    stopwords = set(STOPWORDS)
    for x in stopwords_list: 
        stopwords.add(x.strip('\n'))

    wc = WordCloud(
        background_color='white',
        mask=train_image,
        stopwords=stopwords,
        color_func=image_colors)

    wordcloud = wc.generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a train word cloud')
    parser.add_argument('input', metavar='input', type=str, 
        help='text file containing the words to use to generate')

    args = parser.parse_args()

    run(args.input)