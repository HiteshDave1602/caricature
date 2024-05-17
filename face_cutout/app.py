import io
import cv2
import numpy as np

from io import BytesIO


from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.responses import StreamingResponse
from utils import decode_image,process_face_segmentation,parse_xml_bounding_box,process_face_segmentation_simple
from PIL import Image



app = FastAPI()


# @app.post("/face_segmentation/")
# async def face_segmentation(file: UploadFile = File(...)):
#     try:
#         # Read the uploaded image file
#         contents = await file.read()

#         # Decode the image
#         image = decode_image(contents)

#         # Perform face segmentation
#         face_image = process_face_segmentation(image)

#         # Convert PIL Image to bytes
#         img_byte_array = io.BytesIO()
#         face_image.save(img_byte_array, format="PNG")
#         img_byte_array = img_byte_array.getvalue()

#         # Save the output image locally
#         with open("output_image.png", "wb") as f:
#             f.write(img_byte_array)

#         return StreamingResponse(io.BytesIO(img_byte_array), media_type="image/png")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/face_segmentation/")
async def face_segmentation(file: UploadFile = File(...), xml_file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image_contents = await file.read()
        
        # Read the uploaded XML file
        xml_contents = await xml_file.read()

        # Decode the image
        image = decode_image(image_contents)

        # Parse the bounding box from the XML
        bounding_box = parse_xml_bounding_box(xml_contents)

        # Perform face segmentation
        face_image = process_face_segmentation(image, bounding_box)

        # Convert PIL Image to bytes
        img_byte_array = io.BytesIO()
        face_image.save(img_byte_array, format="PNG")
        img_byte_array = img_byte_array.getvalue()

        return StreamingResponse(io.BytesIO(img_byte_array), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


 

@app.post("/face_segmentation-1/")
async def face_segmentation(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        contents = await file.read()

        # Decode the image
        image = decode_image(contents)

        # Perform face segmentation
        face_image = process_face_segmentation_simple(image)

        # Convert PIL Image to bytes
        img_byte_array = BytesIO()
        face_image.save(img_byte_array, format="PNG")
        img_byte_array = img_byte_array.getvalue()

        return StreamingResponse(io.BytesIO(img_byte_array), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    



def decode_image(image_data):
    try:
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")

def face_segmentation_process(image):
    # Convert RGB image to BGR for OpenCV
    np_image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Load pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert image to grayscale
    gray_image = cv2.cvtColor(np_image_bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        raise ValueError("No faces detected in the image.")

    # Using the first detected face for segmentation
    x, y, w, h = faces[0]

    # Expand the bounding box slightly
    x -= int(w * 0.1)
    y -= int(h * 0.1)
    w += int(w * 0.2)
    h += int(h * 0.2)

    # Apply GrabCut algorithm
    segmented_face = grabcut_algorithm(np_image_bgr, (x, y, w, h))

    return segmented_face

def grabcut_algorithm(original_image, bounding_box):
    segment = np.zeros(original_image.shape[:2], np.uint8)
    x, y, width, height = bounding_box
    segment[y:y+height, x:x+width] = cv2.GC_PR_FGD

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    mask, _, _ = cv2.grabCut(original_image, segment, None, background_model, foreground_model, iterCount=5, mode=cv2.GC_INIT_WITH_MASK)

    # Convert the result to binary mask
    mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

    # Smooth the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply the mask to the original image
    segmented_image = cv2.bitwise_and(original_image, original_image, mask=mask)

    # Convert back to PIL Image
    segmented_image_pil = Image.fromarray(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))

    return segmented_image_pil

@app.post("/face_segmentation-testing/")
async def face_segmentation_endpoint(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        contents = await file.read()

        # Decode the image
        image = decode_image(contents)

        # Perform face segmentation
        face_image = face_segmentation_process(image)  # Await the result

        # Convert PIL Image to bytes
        img_byte_array = io.BytesIO()
        face_image.save(img_byte_array, format="PNG")
        img_byte_array.seek(0)

        return StreamingResponse(img_byte_array, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))