import json
from flask import Flask, render_template, redirect, url_for, request
from flask_wtf import FlaskForm
from requests.exceptions import ConnectionError
import requests
from wtforms import SubmitField
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms.validators import DataRequired
from werkzeug import secure_filename
import urllib.request
import os

class ClientDataForm(FlaskForm):
    file = FileField('File', validators=[FileRequired(), FileAllowed(['avi', 'mp4'], 'Videos only!')])
    submit = SubmitField('Submit')
  
app = Flask(__name__)
app.config.update(
    CSRF_ENABLED=True,
    SECRET_KEY='you-will-never-guess',
)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.config['UPLOAD_PATH'] = '/app/uploads'


def get_prediction(filename):
    body = {'filename': filename}

    myurl = "http://0.0.0.0:8180/predict"
    req = urllib.request.Request(myurl)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(body)
    jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
    req.add_header('Content-Length', len(jsondataasbytes))
    #print (jsondataasbytes)
    try:
        response = urllib.request.urlopen(req, jsondataasbytes)
    except requests.HTTPError as e:
        content = e.read()
    return json.loads(response.read())['predictions']

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predicted/<response>')
def predicted(response):
    response = json.loads(response)
    print(response)
    return render_template('predicted.html', response=response)

@app.route('/predict_form', methods=['GET', 'POST'])
def predict_form():
    form = ClientDataForm()
    data = dict()
    if request.method == 'POST':
        uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file.filename)
        if uploaded_file.filename != '':
            uploaded_file.save(app.config['UPLOAD_PATH'])
        data['filename'] = filename
        try:
            response = str(get_prediction(data['filename']))
            print(response)
        except ConnectionError:
            response = json.dumps({"error": "ConnectionError"})
        return redirect(url_for('predicted', response=response))
    return render_template('form.html', form=form)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8181, debug=True)
