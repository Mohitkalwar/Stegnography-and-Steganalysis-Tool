import numpy as np
import cv2
import joblib  # For loading the model
model_path = 'rf_steganography_model.pkl'
test_image_path = 'asdf.png'
rf_model = joblib.load(model_path)
def preprocess_image(image_path):
    import cv2
    import numpy as np

    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    
    # Resize to 512x512
    image = cv2.resize(image, (512, 512))
    
    # Flatten the image into a single vector
    flattened_image = image.flatten()
    
    # Verify the feature size matches 262,144
    if flattened_image.shape[0] != 262144:
        raise ValueError(f"Expected 262,144 features, but got {flattened_image.shape[0]}!")
    
    return flattened_image

test_image = preprocess_image(test_image_path)

# Reshape the image into the correct input shape for the model
test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension
prediction = rf_model.predict(test_image)

# Output the prediction
if prediction[0] == 0:
    print("The image is predicted to be CLEAN.")
else:
    print("The image is predicted to be STEGANOGRAPHED.")