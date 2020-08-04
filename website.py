import os, cv2
import numpy as np
import random
from keras import backend as K
from keras import optimizers
from keras.layers import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import TimeDistributed, LSTM
from keras.models import Sequential
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['mp4', '3gp'])
UPLOAD_FOLDER = './static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("upload.html")

@app.route('/', methods=['GET', 'POST'])
def home_next():
    K.clear_session()
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        f = request.files['file']
        print(f.filename)

        file = f.filename
        file = file.split(".")[0]

        if f.filename == '':
            print("no uploaded file selected")
            return redirect(request.url)

        if f and allowed_file(f.filename):
            name = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))

        def manual_count(vidcap):
            frames = 0
            while True:
                status, frame = vidcap.read()
                if not status:
                    break
                frames += 1
            return frames

        def convertToImage(data_list):
            min = 10
            img_list = []

            vidcap = cv2.VideoCapture(data_list)
            length = manual_count(vidcap)

            vidcap = cv2.VideoCapture(data_list)
            if (length != min):
                my_list = list(range(1, length - 1))
                lines = random.sample(my_list, min - 2)
                lines.sort()

                success, image = vidcap.read()
                count = 0
                check = 0
                moo = 0
                while success:
                    image = cv2.resize(image, (heightwidth, heightwidth))

                    if (check == 0 or check == length - 1):
                        img_list.append(image)
                        count += 1
                        check += 1
                    elif (check == lines[moo]):
                        img_list.append(image)
                        count += 1
                        check += 1
                        moo += 1
                    else:
                        check += 1

                    if (moo == min - 2):
                        moo = 0
                    success, image = vidcap.read()
                my_list.clear()
                lines.clear()
            else:
                success, image = vidcap.read()
                count = 0
                while success:
                    image = cv2.resize(image, (heightwidth, heightwidth))
                    img_list.append(image)
                    count += 1
                    success, image = vidcap.read()
            return img_list

        num_classes = 26
        time_steps = 10
        heightwidth = 128
        count = 0

        img = convertToImage(name)
        img_data = np.array(img)
        img_data = img_data.astype('float32')
        img_data /= 255

        for x in img:
            cv2.imwrite("static/" + file + "-%d.jpg" % count, x)
            count += 1

        # Make X_test and y_test
        X_test = img_data

        # reshape
        X_test = X_test.reshape(int(X_test.shape[0] / time_steps), time_steps, heightwidth, heightwidth, 3)

        model = Sequential()
        model.add(TimeDistributed(Conv2D(8, (3, 3), strides=(1, 1)), input_shape=X_test.shape[1:]))
        model.add(TimeDistributed(Conv2D(8, (3, 3), strides=(1, 1))))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation("relu")))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        model.add(TimeDistributed(Conv2D(16, (3, 3), strides=(1, 1))))
        model.add(TimeDistributed(Conv2D(16, (3, 3), strides=(1, 1))))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation("relu")))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        model.add(TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1))))
        model.add(TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1))))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation("relu")))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        model.add(Dropout(0.25))
        model.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1))))
        model.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1))))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation("relu")))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        model.add(Dropout(0.25))
        model.add(TimeDistributed(Flatten()))
        model.add(TimeDistributed(Dense(1024)))
        model.add(Dropout(0.5))
        model.add(LSTM(512, return_sequences=False, input_shape=(X_test.shape[2], X_test.shape[3])))
        model.add(Dense(128))
        model.add(Dense(num_classes, activation='softmax'))

        adam = optimizers.Adam(lr=0.00001)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

        fname = "weights1.hdf5"
        model.load_weights(fname)
        test_image = X_test[0:1]
        result = model.predict_classes(test_image)
        result = result[0] + 65
        hasil = str(chr(result))
        return render_template("upload_hasil.php", hasil=hasil, name=name, file=file)
    else:
        return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True)