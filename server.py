#-*- coding: utf-8 -*- 
from flask import *
from flask_compress import Compress
import os
import io
from PIL import Image
import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
import json

class WEB:
    def __init__(self):
        self.imageSize = (128,128)
        self.input_shape = self.imageSize + (3,)
        self.model = tf.keras.models.load_model('model.h5')

    def eval_image(self, img):
        print(img.shape)
        img = cv2.resize(img, self.imageSize, interpolation=cv2.INTER_AREA)
        img0 = img
        img90 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img180 = cv2.rotate(img, cv2.ROTATE_180)
        img270 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = np.array([img0, img90, img180, img270])

        print(img.shape)
        x_eval = img.reshape((4,) + self.input_shape)
        print(x_eval.shape)
        pred = self.model.predict(x_eval)
        
        df = pd.DataFrame(pred, columns=['bang', 'chamdom','mandoong','nong','ozing','woo'])
        df = df.sum(axis=0)
        
        print(type(df))
        
        return df
    

obj = WEB()

compress = Compress()
app = Flask(__name__)
app.secret_key = os.urandom(12)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/load_json', methods=['GET'])
def load_json():
    arg = request.args.get('path')
    target = './json/' + arg + ".json"
    f = open(target, 'r')
    j_file = f.read()   # read file
    f.close()
    return {"status": 200, "message":j_file}, 200

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' in request.files:
        pass
    elif request.files['file'].filename != '':
        pass
    elif 'file' in request.json:
        pass
    else:
        return {"status": 400, "message": "File not found"}, 400
    image = request.files.get('file')
    
    image_binary = image.read()
    img = np.frombuffer(image_binary, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    df = obj.eval_image(img)

    j_df = json.loads(df.to_json(orient='split'))
    print(type(j_df))
    return {"status": 200, "message": j_df}, 200




if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0", threaded=True, port=80)
