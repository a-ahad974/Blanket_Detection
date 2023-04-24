
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def detect_blanket(image_path):
    with_blanket = np.load('with_blanket.npy')
    without_blanket = np.load('without_blanket.npy')

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
    y_pred = svm.predict(x_test)

    haar_data = cv2.CascadeClassifier('data.xml')
    img = cv2.imread(image_path)
    face = haar_data.detectMultiScale(img)

    for x, y, w, h in face:
        user_face = img[y:y + h, x:x + w, :]
        user_face = cv2.resize(user_face, (50, 50))
        user_face = user_face.reshape(1, -1)
        user_face = pca.transform(user_face)
        pred = svm.predict(user_face)
        return names[int(pred)]

    return 'Blanket'


#image_path = 'input_image3.jpg'
#result = detect_blanket(image_path)
#print(result)

