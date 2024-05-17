# import torch
# from PIL import Image
# import numpy as np
# from RealESRGAN import RealESRGAN

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = RealESRGAN(device, scale=4)
# model.load_weights('weights/RealESRGAN_x4.pth', download=True)

# path_to_image = 'wp4031020.png'
# image = Image.open(path_to_image).convert('RGB')

# sr_image = model.predict(image)

# sr_image.save('sr_image1.png')


#  # write a code i am using background image for remove background so now make a code like that i dont want to use that image bt default background should be white



from fastapi import FastAPI, File, UploadFile, HTTPException
from remover import BackgroundRemover
import os
import tempfile
import base64
from PIL import Image
from fastapi.responses import StreamingResponse

app = FastAPI()
obj = BackgroundRemover()

DEFAULT_BACKGROUND_COLOR = (255, 255, 255) 

@app.post("/remove_background/")
async def remove_background(file: UploadFile = File(...)):
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    filename, file_extension = os.path.splitext(file.filename)

    if file_extension.lower() not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, and PNG files are allowed.")

    output_filename = None
    try:
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            file_contents = await file.read()
            temp_file.write(file_contents)

            # Process the image
            result = obj.process_image(temp_file.name, None, default_background=DEFAULT_BACKGROUND_COLOR)

            output_filename = f"processed_{filename}.png"
            result.save(output_filename)

            # Encode the image in base64
            with open(output_filename, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode()

        decoded_image = base64.b64decode(encoded_image)

        decoded_image_filename = f"decoded_{filename}.png"
        with open(decoded_image_filename, "wb") as decoded_file:
            decoded_file.write(decoded_image)

        return StreamingResponse(io.BytesIO(decoded_image), media_type="image/png")  
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file:
            os.unlink(temp_file.name)
        if output_filename:
            os.unlink(output_filename)

from PIL import ImageDraw
from io import BytesIO
import cv2
import numpy as np

def decode_base64_image(image_base64):
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        # Convert to PIL Image
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Invalid image data  {e}")


def process_face_analysis(image):
    # Convert PIL Image to numpy array
    np_image = np.array(image)

    # Convert RGB to BGR (OpenCV uses BGR format)
    np_image_bgr = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

    # Load pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert image to grayscale for face detection
    gray_image = cv2.cvtColor(np_image_bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Convert numpy array back to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(np_image_bgr, cv2.COLOR_BGR2RGB))

    # Draw rectangles around detected faces
    draw = ImageDraw.Draw(pil_image)
    for (x, y, w, h) in faces:
        draw.rectangle([(x, y), (x+w, y+h)], outline="red", width=2)

    return pil_image

@app.post("/face_analysis/")
async def face_analysis(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        contents = await file.read()

        # Convert contents to BytesIO object
        image_data = BytesIO(contents)

        # Open the image using PIL
        image = Image.open(image_data)

        # Perform face analysis
        processed_image = process_face_analysis(image)

        # Save the processed image locally
        processed_image_path = f"processed_{file.filename}"
        processed_image.save(processed_image_path)

        return {"processed_image_path": processed_image_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import shutil
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from PIL import Image
import io
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse
from tempfile import NamedTemporaryFile
import face_recognition


def segment_face(image_path, face_location):
    # Read the uploaded image
    image = np.array(Image.open(image_path))

    top, right, bottom, left = face_location
    face_image = image[top:bottom, left:right]

    gray_face_image = rgb2gray(face_image)

    noiseless_face_image = gaussian(gray_face_image, 1)

    center_x = (right + left) // 2
    center_y = (bottom + top) // 2
    x1 = center_x + 100 * np.cos(np.linspace(0, 2 * np.pi, 500))
    x2 = center_y + 100 * np.sin(np.linspace(0, 2 * np.pi, 500))

    # Generating a circle based on x1, x2
    initial_snake = np.array([x1, x2]).T

    # Compute the active contour for the face region
    segmented_snake = active_contour(noiseless_face_image, initial_snake)

    return segmented_snake

@app.post("/segment-face/")
async def segment_face_endpoint(file: UploadFile = File(...)):
    # Save the uploaded image temporarily
    with NamedTemporaryFile(delete=False, suffix=".png") as temp:
        temp.write(await file.read())
        temp_image_path = temp.name

    image = face_recognition.load_image_file(temp_image_path)
    face_locations = face_recognition.face_locations(image)

    if not face_locations:
        return {"message": "No faces found in the image."}

    face_location = face_locations[0]

    # Segment the face
    segmented_snake = segment_face(temp_image_path, face_location)

    segmented_image = np.array(Image.open(temp_image_path))
    segmented_image[segmented_snake[:, 1].astype(int), segmented_snake[:, 0].astype(int)] = [255, 0, 0]  # Red color

    segmented_image_path = temp_image_path.replace(".png", "_segmented.png")
    Image.fromarray(segmented_image).save(segmented_image_path)

    return FileResponse(segmented_image_path, media_type="image/png")



# import cv2 
# import os
# from fastapi import FastAPI, UploadFile, File
# from tempfile import NamedTemporaryFile
# import numpy as np


# app = FastAPI()

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def detect_and_crop_faces(image):
#     image_bytes = image.file.read()
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
#     gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
#     cropped_faces_paths = []
#     for (x, y, w, h) in faces:
#         face = img_np[y:y+h, x:x+w]
#         save_path = f"cropped_face_{x}_{y}.png"
#         cv2.imwrite(save_path, face)
#         cropped_faces_paths.append(save_path)
        
#     return cropped_faces_paths

# @app.post("/detect-face/")
# async def detect_face(image: UploadFile = File(...)):
#     cropped_faces_paths = await detect_and_enhance_face(image)
#     return cropped_faces_paths

# import cv2
# import numpy as np
# import mediapipe as mp
# import base64

# from fastapi import UploadFile, HTTPException
# from fastapi.responses import JSONResponse


# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# mp_face_detection = mp.solutions.face_detection


# async def detect_and_enhance_face(file: UploadFile):
#     if file.content_type.startswith('image/'):
#         contents = await file.read()

#         nparr = np.frombuffer(contents, np.uint8)
#         image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         found_faces = False  # Flag to track if faces were found
#         if image is not None:  # Check if image is not None
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#             cropped_faces_paths = []
#             for (x, y, w, h) in faces:
#                 face = image[y:y+h, x:x+w]
#                 save_path = f"cropped_face_{x}_{y}.png"
#                 cv2.imwrite(save_path, face)
#                 cropped_faces_paths.append(save_path)

#             if cropped_faces_paths:
#                 found_faces = True
#             else:
#                 with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
#                     results = face_detection.process(image)

#                 if results.detections:
#                     encoded_faces = []
#                     decoded_faces = []

#                     for i, detection in enumerate(results.detections):
#                         bboxC = detection.location_data.relative_bounding_box
#                         ih, iw, _ = image.shape
#                         bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
#                         face = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

#                         _, buffer = cv2.imencode('.jpg', face)
#                         encoded_face = base64.b64encode(buffer).decode()
#                         encoded_faces.append(encoded_face)

#                         filename = f"face_{i}.jpg"
#                         decoded_faces.append(filename)

#                     return JSONResponse(content={"encoded_faces": encoded_faces, "decoded_faces": decoded_faces})
#                 else:
#                     raise HTTPException(status_code=400, detail="No faces detected in the provided image.")
#         else:
#             raise HTTPException(status_code=400, detail="Error processing the image.")

#         if found_faces:
#             return cropped_faces_paths
#         else:
#             raise HTTPException(status_code=400, detail="No faces detected in the provided image.")
#     else:
#         raise HTTPException(status_code=415, detail="Unsupported file type, only images are allowed.")


#  why not saving image in locally 


# def detect_and_crop_faces(image):
#     image_bytes = image.file.read()
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     cropped_faces_paths = []
#     for (x, y, w, h) in faces:
#         face = img_np[y:y+h, x:x+w]
#         save_path = f"cropped_face_{x}_{y}.png"
#         cv2.imwrite(save_path, face)
#         cropped_faces_paths.append(save_path)

#     return cropped_faces_paths



