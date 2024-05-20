import os
import cv2
import base64
import numpy as np
from PIL import Image, ImageEnhance
import mediapipe as mp
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse


import torch

model_path = "cartoonish_v1.safetensors"

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
try:
    model = torch.load(model_path, map_location=device)
except Exception as e:
    print(f"Error loading model: {e}")



import os
import cv2
import numpy as np
import base64
from PIL import Image, ImageEnhance
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse
import mediapipe as mp



import cv2
import numpy as np
import mediapipe as mp
from PIL import ImageEnhance
import os
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import base64

import torch
import torchvision.transforms as transforms
import io

from diffusers import StableDiffusionInstructPix2PixPipeline

# 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_face_detection = mp.solutions.face_detection


async def detect_and_enhance_face(file: UploadFile):
    if file.content_type.startswith('image/'):
        contents = await file.read()

        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Error decoding the image.")

        found_face = False

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        cropped_faces_paths = []
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            save_path = f"cropped_with_cv_{x}_{y}.png"
            cv2.imwrite(save_path, face)
            cropped_faces_paths.append(save_path)

        if cropped_faces_paths:
            found_face = True
        else:
            mp_face_detection = mp.solutions.face_detection
            with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                results = face_detection.process(image)

            if results.detections:
                encoded_faces = []
                decoded_faces = []
                cartoon_faces_encode = []
                cartoon_faces_decode = []

                for i, detection in enumerate(results.detections):
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    face = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

                    blur_intensity = min(face.shape[0], face.shape[1]) / max(image.shape[0], image.shape[1])
                    blur_radius = max(1, int(blur_intensity * 10))
                    blur_radius += 1 if blur_radius % 2 == 0 else 0
                    face = cv2.GaussianBlur(face, (blur_radius, blur_radius), 0)

                    cartoon_image, enhanced_cartoon_path = await convert_to_cartoon(face)
                    cartoon_image_str = base64.b64encode(cv2.imencode('.jpg', cartoon_image)[1]).decode('utf-8')
                    cartoon_faces_encode.append(cartoon_image_str)

                    cartoon_image_filename = f"cartoon_face_{i}.jpg"
                    cartoon_faces_decode.append(cartoon_image_filename)

                    _, buffer = cv2.imencode('.jpg', face)
                    encoded_face = base64.b64encode(buffer).decode()
                    encoded_faces.append(encoded_face)

                    filename = f"face_{i}.jpg"
                    decoded_faces.append(filename)

                return JSONResponse(content={"encoded_faces": encoded_faces, "decoded_faces": decoded_faces, "cartoon_faces": cartoon_faces_encode, "enhanced_cartoon_path": enhanced_cartoon_path})

        if found_face:
            return cropped_faces_paths
        else:
            raise HTTPException(status_code=400, detail="No faces detected in the provided image.")
    else:
        raise HTTPException(status_code=415, detail="Unsupported file type, only images are allowed.")



model_id = "instruction-tuning-sd/scratch-cartoonizer"
pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, use_auth_token=True
).to("cuda") 

def preprocess_image(image):
    """Preprocesses an image for use with the diffusion model.

    Args:
        image: The image as a PIL Image object.

    Returns:
        A preprocessed image tensor.
    """

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    return preprocess(image).unsqueeze(0)

import io

async def convert_to_cartoon(face_image):
    """Converts a face image to a cartoon using a diffusion model.

    Args:
        face_image: NumPy array representing the face image.

    Returns:
        The cartoonized face image as a NumPy array.
    """
    try:
        # Convert NumPy array to PIL image
        pil_image = Image.fromarray(face_image)

        # Preprocess the image
        input_tensor = preprocess_image(pil_image)

        # Create a prompt for the diffusion model
        input_prompt = "Convert this image into a cartoon"

        # Perform model inference (disable gradients for efficiency)
        with torch.no_grad():
            cartoon_image_list = pipeline(prompt=input_prompt, image=input_tensor)

        # Extract the first generated image (assuming single output)
        output_image = cartoon_image_list.images[0]

        # Convert PIL image to NumPy array
        cartoon_image = np.array(output_image)

        enhanced_cartoon_path = await enhance_cartoon(cartoon_image) 

        return cartoon_image,enhanced_cartoon_path
    except Exception as e:
        print(f"Error converting image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during conversion.")



# async def convert_to_cartoon(file: UploadFile = File(...)):
#     """Converts a user-uploaded image to a cartoon using a diffusion model.

#     Args:
#         file: The uploaded image file (expects a valid image format).

#     Returns:
#         A StreamingResponse containing the generated cartoon image as JPEG data.

#     Raises:
#         HTTPException: If the uploaded file is not an image.
#     """

#     if file.content_type.startswith('image/'):
#         try:
#             # Open image from bytes
#             pil_image = Image.open(io.BytesIO(await file.read()))

#             # Preprocess the image
#             input_tensor = preprocess_image(pil_image)

#             # Create a prompt for the diffusion model
#             input_prompt = "Convert this image into a cartoon"

#             # Perform model inference (disable gradients for efficiency)
#             with torch.no_grad():
#                 cartoon_image_list = pipeline(prompt=input_prompt, image=input_tensor)

#             # Extract the first generated image (assuming single output)
#             output_image = cartoon_image_list.images[0]

#             # Save the image locally
#             output_image_path = "cartoon_output.jpg"  # Choose your desired path and filename
#             output_image.save(output_image_path, format='JPEG')

#             enhanced_image_path = await enhance_cartoon(output_image_path)

#             if enhanced_image_path:

#                 # Function to create a streaming response for the cartoon image
#                 async def image_stream():
#                     with open(output_image_path, "rb") as f:
#                         while True:
#                             chunk = f.read(1024)
#                             if not chunk:
#                                 break
#                             yield chunk

#             # Return the streaming response
#             return StreamingResponse(image_stream(), media_type="image/jpeg")
#         except Exception as e:
#             print(f"Error converting image: {e}")
#             raise HTTPException(status_code=500, detail="Internal server error during conversion.")

#     else:
#         raise HTTPException(status_code=415, detail="Unsupported file type, only images are allowed.")





import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device, scale=3)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

async def enhance_cartoon(cartoon_image):
    try:
        sr_cartoon_image = model.predict(Image.fromarray(cartoon_image))

        enhanced_cartoon_path = f"enhanced_cartoon_face_{os.urandom(4).hex()}.jpg"
        sr_cartoon_image.save(enhanced_cartoon_path, format='JPEG')

        return enhanced_cartoon_path
    except Exception as e:
        print(f"Error enhancing image: {e}")
        return None


def calculate_brightness_factor(face):
    mean_brightness = np.mean(face)
    brightness_factor = 1.0 + (mean_brightness - 128) / 128
    return brightness_factor


def calculate_contrast_factor(face):
    std_dev = np.std(face)
    contrast_factor = 1.0 + (std_dev - 64) / 64
    return contrast_factor

