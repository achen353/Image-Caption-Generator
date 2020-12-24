from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse


def extract_features(filename, model):
    print("File Path: \"" + filename + "\"")

    try:
        image = cv2.imread(filename, -1)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct.")

    image = cv2.resize(image, (299, 299))
    image = np.array(image)
    # For images that has 4 channels, we convert them into 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


def main():
    # Example image: 'flickr8k-dataset/111537222_07e56d5a30.jpg'
    # Command: python testing_caption_generator.py -i ./flickr8k-dataset/111537222_07e56d5a30.jpg
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required=True, help="Image Path")
    args = vars(parser.parse_args())
    img_path = args['image']

    max_length = 32
    tokenizer = load(open("tokenizer.p","rb"))
    model = load_model('models/model_9.h5')
    xception_model = Xception(include_top=False, pooling="avg")

    photo = extract_features(img_path, xception_model)
    img = cv2.imread(img_path, 0)

    description = generate_desc(model, tokenizer, photo, max_length)

    if description != 'start':
        description = description[6:]
    if description[-3:] == 'end':
        description = description[:-3]

    print("\n")
    print("Caption: " + description)
    plt.imshow(img)

if __name__ == '__main__':
    main()
