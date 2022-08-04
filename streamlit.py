# Developed by Yash kumar Jain
# The model has been taken from the jupyter notebook which I developed during training phase, so it will reduce time.

from keras.saving.model_config import model_from_json
import streamlit as st
from PIL import Image
from skimage.util.shape import view_as_blocks
from skimage import io, transform
import warnings

piece_symbols = 'prbnkqPRBNKQ'
warnings.filterwarnings('ignore')


def load_image(image_file):
    img = Image.open(image_file)
    return img


def process_image(img):
    downsample_size = 200
    square_size = int(downsample_size / 8)
    img_read = io.imread(img)
    img_read = transform.resize(img_read, (downsample_size, downsample_size), mode='constant')
    tiles = view_as_blocks(img_read, block_shape=(square_size, square_size, 3))
    tiles = tiles.squeeze(axis=2)
    return tiles.reshape(64, square_size, square_size, 3)


def image_onehot(one_hot):
    output = ''
    for j in range(8):
        for i in range(8):
            if one_hot[j][i] == 12:
                output += ' '
            else:
                output += piece_symbols[one_hot[j][i]]
        if j != 7:
            output += '-'
    for i in range(8, 0, -1):
        output = output.replace(' ' * i, str(i))
    return output


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")


def main():
    st.subheader("Upload Chess board Image")
    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
    if st.button('Predict'):
        st.image(load_image(image_file), width=250)
        pred = loaded_model.predict(process_image(image_file)).argmax(axis=1).reshape(-1, 8, 8)
        prediction = image_onehot(pred[0])
        st.write('Predicted:', prediction)


if __name__ == '__main__':
    main()

