import glob
import time
import pickle
import os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from . import lesson_functions


def read_images():
    """
    Read the training images 
    Returns the images splited between cars and not cars

    This test images are downloaded from the  GTI vehicle image database
    http://www.gti.ssr.upm.es/data/Vehicle_database.html
    """
    # Read in car and non-car images
    images = glob.glob('images/**/**/*.png')
    cars = []
    notcars = []
    for image in images:
        if 'non-vehicles' in image:
            notcars.append(image)
        else:
            cars.append(image)

    print("Cars found:     ", len(cars))
    print("Not cars found: ", len(notcars))
    print("Total:         ", len(cars) + len(notcars))

    return cars, notcars


def training():
    """
    Training Linear SVC
    Returns the trained Linear SVC and a trained StandardScaler 
    """

    dir = os.path.dirname(__file__)
    trainingFilePath =  dir + "/../../training.p"

    if os.path.isfile(trainingFilePath) is False:
        t = time.time()

        cars, notcars = read_images()

        print("Extracting features...")
        car_features = lesson_functions.extract_features(cars)
        notcar_features = lesson_functions.extract_features(notcars)


        print("Getting vectors...")
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        X_scaled = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


        # Split up data into randomized training and test sets
        print("Splitting...")
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        print(len(y_train), "training images")
        print(len(y_test), "testing images")


        # Use a linear SVC 
        print("Training...")
        svc = LinearSVC()
        svc.fit(X_train, y_train)

        # Get the score of the SVC
        test_accuracy = round(svc.score(X_test, y_test), 4)
        print('Test Accuracy of SVC = ', test_accuracy)

        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train')

        # save values to transform 3D to 2D
        data = {'svc': svc, 'X_scaler': X_scaler}

        # save file
        with open(trainingFilePath, 'wb') as f:
            pickle.dump(data, file=f)

    else:
        trainingData = pickle.load( open(trainingFilePath, "rb") )
        svc = trainingData['svc']
        X_scaler = trainingData['X_scaler']


    return svc, X_scaler
