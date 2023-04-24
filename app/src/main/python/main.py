import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import base64


def detect_blanket(image_string):
    
    file_dir = os.path.dirname(os.path.realpath(__file__))
    with_blanket_path = os.path.join(file_dir, 'with_blanket.npy')
    without_blanket_path = os.path.join(file_dir, 'without_blanket.npy')

    with_blanket = np.load(with_blanket_path)
    without_blanket = np.load(without_blanket_path)	

    with_blanket = with_blanket.reshape(200, 50 * 50 * 3)
    without_blanket = without_blanket.reshape(200, 50 * 50 * 3)
		
    x = np.r_[with_blanket, without_blanket]

    labels = np.zeros(x.shape[0])
    labels[200:] = 1.0
    names = {0: 'Blanket', 1: 'No Blanket'}

    x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.20)

    pca = PCA(n_components=3)
    x_train = pca.fit_transform(x_train)

    svm = SVC()
    svm.fit(x_train, y_train)

    x_test = pca.transform(x_test)

    # Decode the base64-encoded string to get the byte array
    image_bytes = base64.b64decode(image_string)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    data_xml_path = os.path.join(file_dir, 'data.xml')
    haar_data = cv2.CascadeClassifier(data_xml_path)
    face = haar_data.detectMultiScale(img)

    for x, y, w, h in face:
        user_face = img[y:y + h, x:x + w, :]
        user_face = cv2.resize(user_face, (50, 50))
        user_face = user_face.reshape(1, -1)
        user_face = pca.transform(user_face)
        pred = svm.predict(user_face)
        return names[int(pred)]

    return 'Blanket'

