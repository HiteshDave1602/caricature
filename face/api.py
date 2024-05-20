from fastapi import APIRouter,UploadFile,File

from face import crud

router = APIRouter(
    prefix="/Face_detection",
    tags=["cutout_face"]
)

@router.post("/crop_faces/")
async def crop_faces(file: UploadFile = File(...)):
    crop_faces = await crud.detect_and_enhance_face(file)

    return crop_faces


from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline, StableDiffusionInstructPix2PixPipeline
import io

# Load the pre-trained model with proper authentication
model_id = "instruction-tuning-sd/scratch-cartoonizer"
pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, use_auth_token=True
).to("cuda") 

# Define the image preprocessing function
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

# Create the FastAPI application

import os

@router.post("/convert-to-cartoon")
async def convert_to_cartoon(file: UploadFile = File(...)):
    """Converts a user-uploaded image to a cartoon using a diffusion model.

    Args:
        file: The uploaded image file (expects a valid image format).

    Returns:
        A StreamingResponse containing the generated cartoon image as JPEG data.

    Raises:
        HTTPException: If the uploaded file is not an image.
    """

    if file.content_type.startswith('image/'):
        try:
            # Open image from bytes
            pil_image = Image.open(io.BytesIO(await file.read()))

            # Preprocess the image
            input_tensor = preprocess_image(pil_image)

            # Create a prompt for the diffusion model
            input_prompt = "Convert this image into a cartoon"

            # Perform model inference (disable gradients for efficiency)
            with torch.no_grad():
                cartoon_image_list = pipeline(prompt=input_prompt, image=input_tensor)

            # Extract the first generated image (assuming single output)
            output_image = cartoon_image_list.images[0]

            # Save the image locally
            output_image_path = "cartoon_output.jpg"  # Choose your desired path and filename
            output_image.save(output_image_path, format='JPEG')

            # Function to create a streaming response for the cartoon image
            async def image_stream():
                with open(output_image_path, "rb") as f:
                    while True:
                        chunk = f.read(1024)
                        if not chunk:
                            break
                        yield chunk

            # Return the streaming response
            return StreamingResponse(image_stream(), media_type="image/jpeg")
        except Exception as e:
            print(f"Error converting image: {e}")
            raise HTTPException(status_code=500, detail="Internal server error during conversion.")

    else:
        raise HTTPException(status_code=415, detail="Unsupported file type, only images are allowed.")



# @router.post("/convert-to-cartoon")
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

#             # Function to create a streaming response for the cartoon image
#             async def image_stream():
#                 image_bytes = io.BytesIO()
#                 output_image.save(image_bytes, format='JPEG')
#                 image_bytes.seek(0)
#                 yield image_bytes.getvalue()

#             # Return the streaming response
#             return StreamingResponse(image_stream(), media_type="image/jpeg")
#         except Exception as e:
#             print(f"Error converting image: {e}")
#             raise HTTPException(status_code=500, detail="Internal server error during conversion.")

#     else:
#         raise HTTPException(status_code=415, detail="Unsupported file type, only images are allowed.")


from PIL import Image, ImageOps
import cv2
import numpy as np
import mediapipe as mp
import base64
import os
from fastapi import UploadFile, HTTPException, FastAPI
from fastapi.responses import JSONResponse




def auto_contrast(image):
    return ImageOps.autocontrast(image)


@router.post("/face")
async def detect_and_enhance_face(file: UploadFile):
    if file.content_type.startswith('image/'):
        contents = await file.read()

        # Load the input image from bytes
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Initialize Mediapipe face detector
        mp_face_detection = mp.solutions.face_detection

        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(image)

        if results.detections:
            # Directory to save the images
            save_dir = "saved_faces"
            os.makedirs(save_dir, exist_ok=True)

            encoded_faces = []
            decoded_faces = []

            for i, detection in enumerate(results.detections):
                # Extract face bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

                # Convert to PIL Image
                pil_image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

                # Autocontrast and Auto Brightness Adjustment
                adjusted_face = auto_contrast(pil_image)

                # Convert back to OpenCV format
                enhanced_face = cv2.cvtColor(np.array(adjusted_face), cv2.COLOR_RGB2BGR)

                # Enhance and Resize
                # (You can add additional enhancements here)

                # Calculate dynamic resolution maintaining aspect ratio
                scale_percent = 100  # percentage of original size
                width = int(enhanced_face.shape[1] * scale_percent / 100)
                height = int(enhanced_face.shape[0] * scale_percent / 100)
                dim = (width, height)
                resized_face = cv2.resize(enhanced_face, dim, interpolation=cv2.INTER_AREA)

                # Encode upscaled face to base64 string
                _, buffer = cv2.imencode('.jpg', resized_face)
                encoded_face = base64.b64encode(buffer).decode()
                encoded_faces.append(encoded_face)

                # Generate a unique filename for the new image
                filename = f"face_{i}.jpg"
                filepath = os.path.join(save_dir, filename)

                # Save upscaled face locally
                cv2.imwrite(filepath, resized_face)
                decoded_faces.append(filepath)


        else:
            raise HTTPException(status_code=400, detail="No faces detected in the provided image.")
    else:
        raise HTTPException(status_code=415, detail="Unsupported file type, only images are allowed.")
