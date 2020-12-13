import json
import numpy.core.multiarray
import torch
import torchvision
import torchvision.transforms as transforms
import logging
from logging.handlers import RotatingFileHandler
from PIL import Image
from flask import Flask, jsonify, request
import io
import cv2
import argparse
import time
import numpy as np
from imutils.video import FPS
import dill
import pandas as pd
import os
from time import strftime

app = Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def load_model(model_path):
    global model
    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.utils.model_zoo.load_url(model_path)
    return model



class_index = json.load(open('/app/app/classes/classes.json'))

def get_frames(file):
    cap = cv2.VideoCapture(file)
    clips = []
    if not cap.isOpened():
        print('Error while trying to read video. Please check path again')

    fps = FPS().start()
    while True:
        # capture each frame of the video
        (ret, frame) = cap.read()
        if not ret:
            break
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)

        clips.append(pil_im)
        cv2.waitKey(1)
        fps.update()

    fps.stop()
    cap.release()
    # close all frames and video windows
    cv2.destroyAllWindows()

    # input_frames = np.array(clips)
    return clips


class ImglistToTensor(torch.nn.Module):
    """
    Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
    Can be used as first transform for ``VideoFrameDataset``.
    """

    def forward(self, img_list):
        """
        Converts each PIL image in a list to
        a torch Tensor and stacks them into
        a single tensor.
        Args:
            img_list: list of PIL images.
        Returns:
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])


def transform_frames(image_list):
    my_transforms = transforms.Compose([
        ImglistToTensor(),
        transforms.Resize(112),
        transforms.CenterCrop(112),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return my_transforms(image_list).unsqueeze(0)


def get_prediction(file):
    modelpath = 'https://drive.google.com/file/d/1IpTNWbotymVAGL6L29S2zhdT2aqgK4A5/view?usp=sharing'
    model = load_model(modelpath)
    model.eval()
    frames_list = get_frames(file)
    frames_list_new = frames_list[::(int(len(frames_list) / 19) + 1)]
    frames = transform_frames(image_list=frames_list_new)
    inputs = frames.permute(0, 2, 1, 3, 4).to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    predicted_idx = str(preds.item())
    return predicted_idx, class_index[predicted_idx]
    # return frames_list,frames_list_new


@app.route("/", methods=["GET"])
def general():
    return """Welcome to video classification process. Please use 'http://<address>/predict' to POST"""


@app.route('/predict', methods=['POST'])
def predict():
    data = {"success": False}
    dt = strftime("[%Y-%b-%d %H:%M:%S]")
    if request.method == 'POST':
        filename = ""
        request_json = request.get_json()
        # file = request.files['file']
        # filename = '4031844.avi'
        if request_json["filename"]:
            filename = request_json['filename']

        logger.info(f'{dt} Data: filename={filename}')
        try:
            class_id, class_name = get_prediction(file=filename)

        except AttributeError as e:
            logger.warning(f'{dt} Exception: {str(e)}')
            data['predictions'] = str(e)
            data['success'] = False
            return jsonify(data)

        data["predictions"] = class_name
            # indicate that the request was a success
        data["success"] = True

        # return the data dictionary as a JSON response
    return jsonify(data)


if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
           "please wait until server has fully started"))
    port = int(os.environ.get('PORT', 8180))
    app.run(host='0.0.0.0', debug=True, port=port)
