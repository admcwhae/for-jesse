from tracemalloc import stop
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import STOPWORDS, WordCloud, ImageColorGenerator
from PIL import Image

if __name__ == '__main__':
    with open('input.txt') as f:
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
