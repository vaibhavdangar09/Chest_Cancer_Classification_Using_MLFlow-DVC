import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import shutil


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        destination_directory="model"
        file_path=r"artifacts/training/model.h5"
        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)
        shutil.copy(file_path, destination_directory)
        print("Model copied to Model directory")
      
        model = load_model(os.path.join("model", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = 'Dog'
            return [{"image": prediction}]
        else:
            prediction = 'Cat'
            return [{"image": prediction}]