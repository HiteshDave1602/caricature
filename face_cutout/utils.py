# import cv2
# import io
# import numpy as np
# from io import BytesIO
# from PIL import Image



# def decode_image(image_data):
#     try:
#         image = Image.open(BytesIO(image_data))
#         return image
#     except Exception as e:
#         raise ValueError(f"Invalid image data: {e}")

# def process_face_segmentation(image):
#     np_image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     gray_image = cv2.cvtColor(np_image_bgr, cv2.COLOR_BGR2GRAY)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     if len(faces) == 0:
#         raise ValueError("No faces detected in the image.")

#     x, y, w, h = faces[0]

#     x -= int(w * 0.1)
#     y -= int(h * 0.1)
#     w += int(w * 0.2)
#     h += int(h * 0.2)

#     # Apply GrabCut algorithm
#     segmented_face = grabcut_algorithm(np_image_bgr, (x, y, w, h))

#     return segmented_face

# def grabcut_algorithm(original_image, bounding_box):
#     segment = np.zeros(original_image.shape[:2], np.uint8)
#     x, y, width, height = bounding_box
#     segment[y:y+height, x:x+width] = cv2.GC_PR_FGD

#     background_model = np.zeros((1, 65), np.float64)
#     foreground_model = np.zeros((1, 65), np.float64)

#     mask, _, _ = cv2.grabCut(original_image, segment, None, background_model, foreground_model, iterCount=5, mode=cv2.GC_INIT_WITH_MASK)

#     mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     segmented_image = cv2.bitwise_and(original_image, original_image, mask=mask)

#     segmented_image_pil = Image.fromarray(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))

#     return segmented_image_pil


#  write a code i have one xml file in that bounding_box based on that do face segmentation 

import cv2
import numpy as np
from PIL import Image
from io import BytesIO

def decode_image(image_data):
    try:
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")
    

def process_face_segmentation_simple(image):
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


# def process_face_segmentation_simple(image):
#     np_image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     gray_image = cv2.cvtColor(np_image_bgr, cv2.COLOR_BGR2GRAY)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     if len(faces) == 0:
#         raise ValueError("No faces detected in the image.")

#     x, y, w, h = faces[0]

#     # Expand the bounding box to include more facial features
#     padding = 0.2
#     x -= int(w * padding)
#     y -= int(h * padding)
#     w += int(w * padding * 2)
#     h += int(h * padding * 2)

#     # Apply GrabCut algorithm
#     segmented_face = grabcut_algorithm(np_image_bgr, (x, y, w, h))

#     return segmented_face

def grabcut_algorithm(original_image, bounding_box):
    segment = np.zeros(original_image.shape[:2], np.uint8)
    x, y, width, height = bounding_box
    segment[y:y+height, x:x+width] = cv2.GC_PR_FGD

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    mask, _, _ = cv2.grabCut(original_image, segment, None, background_model, foreground_model, iterCount=5, mode=cv2.GC_INIT_WITH_MASK)

    mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    segmented_image = cv2.bitwise_and(original_image, original_image, mask=mask)

    # Fill the background with white color
    segmented_image[np.where((segmented_image == [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    segmented_image_pil = Image.fromarray(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))

    return segmented_image_pil


# 


# import cv2
# import numpy as np
# from PIL import Image
# from io import BytesIO
# import xml.etree.ElementTree as ET


# def decode_image(image_data):
#     try:
#         image = Image.open(BytesIO(image_data))
#         return image
#     except Exception as e:
#         raise ValueError(f"Invalid image data: {e}")

# def parse_xml_bounding_box(xml_data):
#     try:
#         root = ET.fromstring(xml_data)
#         object_element = root.find('object')
#         if object_element is None:
#             raise ValueError("Object element not found in the XML data")
        
#         bndbox = object_element.find('bndbox')
#         if bndbox is None:
#             raise ValueError("Bounding box element not found in the XML data")
        
#         xmin = bndbox.find('xmin')
#         ymin = bndbox.find('ymin')
#         xmax = bndbox.find('xmax')
#         ymax = bndbox.find('ymax')
        
#         if None in (xmin, ymin, xmax, ymax):
#             raise ValueError("Incomplete bounding box information in the XML data")
        
#         x = int(xmin.text)
#         y = int(ymin.text)
#         width = int(xmax.text) - x
#         height = int(ymax.text) - y
        
#         return x, y, width, height
#     except ET.ParseError as e:
#         raise ValueError(f"Error parsing XML data: {e}")
#     except Exception as e:
#         raise ValueError(f"Invalid XML data: {e}")

# def process_face_segmentation(image, bounding_box):
#     np_image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#     x, y, w, h = bounding_box

#     # Ensure the bounding box is within the image dimensions
#     h_img, w_img = np_image_bgr.shape[:2]
#     x = max(0, x)
#     y = max(0, y)
#     w = min(w, w_img - x)
#     h = min(h, h_img - y)

#     # Apply padding to the bounding box
#     padding = 0.2
#     x_padded = max(0, x - int(w * padding))
#     y_padded = max(0, y - int(h * padding))
#     w_padded = min(w_img - x_padded, w + int(w * padding * 2))
#     h_padded = min(h_img - y_padded, h + int(h * padding * 2))

#     # Apply GrabCut algorithm
#     segmented_face = grabcut_algorithm(np_image_bgr, (x_padded, y_padded, w_padded, h_padded))

#     return segmented_face

# def grabcut_algorithm(original_image, bounding_box):
#     x, y, w, h = bounding_box

#     # Create a mask initialized with probable background
#     mask = np.zeros(original_image.shape[:2], np.uint8)
#     mask[:] = cv2.GC_PR_BGD

#     # Set the bounding box region as probable foreground
#     mask[y:y+h, x:x+w] = cv2.GC_PR_FGD

#     # Define a smaller region within the bounding box as sure foreground
#     mask[y+int(0.1*h):y+int(0.9*h), x+int(0.1*w):x+int(0.9*w)] = cv2.GC_FGD

#     # Initialize the background and foreground models
#     background_model = np.zeros((1, 65), np.float64)
#     foreground_model = np.zeros((1, 65), np.float64)

#     # Run GrabCut algorithm
#     cv2.grabCut(original_image, mask, None, background_model, foreground_model, iterCount=5, mode=cv2.GC_INIT_WITH_MASK)

#     # Extract the foreground mask
#     mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

#     # Apply morphological operations to refine the mask
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     # Apply the mask to the original image
#     segmented_image = cv2.bitwise_and(original_image, original_image, mask=mask)

#     # Create a white background
#     white_bg = np.ones_like(original_image, dtype=np.uint8) * 255
#     inv_mask = cv2.bitwise_not(mask)

#     # Combine the segmented image with the white background
#     segmented_image = cv2.add(segmented_image, cv2.bitwise_and(white_bg, white_bg, mask=inv_mask))

#     # Convert the segmented image to PIL format
#     segmented_image_pil = Image.fromarray(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))

#     return segmented_image_pil



import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import xml.etree.ElementTree as ET

def decode_image(image_data):
    try:
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")

def parse_xml_bounding_box(xml_data):
    try:
        root = ET.fromstring(xml_data)
        object_element = root.find('object')
        if object_element is None:
            raise ValueError("Object element not found in the XML data")
        
        bndbox = object_element.find('bndbox')
        if bndbox is None:
            raise ValueError("Bounding box element not found in the XML data")
        
        xmin = bndbox.find('xmin')
        ymin = bndbox.find('ymin')
        xmax = bndbox.find('xmax')
        ymax = bndbox.find('ymax')
        
        if None in (xmin, ymin, xmax, ymax):
            raise ValueError("Incomplete bounding box information in the XML data")
        
        x = int(xmin.text)
        y = int(ymin.text)
        width = int(xmax.text) - x
        height = int(ymax.text) - y
        
        return x, y, width, height
    except ET.ParseError as e:
        raise ValueError(f"Error parsing XML data: {e}")
    except Exception as e:
        raise ValueError(f"Invalid XML data: {e}")

def process_face_segmentation(image, bounding_box):
    np_image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    x, y, w, h = bounding_box

    # Ensure the bounding box is within the image dimensions
    h_img, w_img = np_image_bgr.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)

    # Apply padding to the bounding box
    padding = 0.2
    x_padded = max(0, x - int(w * padding))
    y_padded = max(0, y - int(h * padding))
    w_padded = min(w_img - x_padded, w + int(w * padding * 2))
    h_padded = min(h_img - y_padded, h + int(h * padding * 2))

    # Apply GrabCut algorithm
    segmented_face = grabcut_algorithm(np_image_bgr, (x_padded, y_padded, w_padded, h_padded))

    return segmented_face

def grabcut_algorithm(original_image, bounding_box):
    x, y, w, h = bounding_box

    # Create a mask initialized with probable background
    mask = np.zeros(original_image.shape[:2], np.uint8)
    mask[:] = cv2.GC_PR_BGD

    # Set the bounding box region as probable foreground
    mask[y:y+h, x:x+w] = cv2.GC_PR_FGD

    # Initialize the background and foreground models
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    # Run GrabCut algorithm
    cv2.grabCut(original_image, mask, (x, y, w, h), background_model, foreground_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

    # Extract the foreground mask
    mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

    # Refine the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Extract the segmented face region
    segmented_face = cv2.bitwise_and(original_image, original_image, mask=mask)

    # Convert the segmented face to PIL format
    segmented_face_pil = Image.fromarray(cv2.cvtColor(segmented_face, cv2.COLOR_BGR2RGB))

    # Save the segmented face
    segmented_face_pil.save("segmented_face.png")

    return segmented_face_pil



# import cv2
# import numpy as np
# from PIL import Image
# from io import BytesIO
# import xml.etree.ElementTree as ET

# def decode_image(image_data):
#     try:
#         image = Image.open(BytesIO(image_data))
#         return image
#     except Exception as e:
#         raise ValueError(f"Invalid image data: {e}")

# def parse_xml_bounding_box(xml_data):
#     try:
#         root = ET.fromstring(xml_data)
#         object_element = root.find('object')
#         if object_element is None:
#             raise ValueError("Object element not found in the XML data")
        
#         bndbox = object_element.find('bndbox')
#         if bndbox is None:
#             raise ValueError("Bounding box element not found in the XML data")
        
#         xmin = bndbox.find('xmin')
#         ymin = bndbox.find('ymin')
#         xmax = bndbox.find('xmax')
#         ymax = bndbox.find('ymax')
        
#         if None in (xmin, ymin, xmax, ymax):
#             raise ValueError("Incomplete bounding box information in the XML data")
        
#         x = int(xmin.text)
#         y = int(ymin.text)
#         width = int(xmax.text) - x
#         height = int(ymax.text) - y
        
#         return x, y, width, height
#     except ET.ParseError as e:
#         raise ValueError(f"Error parsing XML data: {e}")
#     except Exception as e:
#         raise ValueError(f"Invalid XML data: {e}")

# def process_face_segmentation(image, bounding_box):
#     np_image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#     x, y, w, h = bounding_box

#     # Ensure the bounding box is within the image dimensions
#     h_img, w_img = np_image_bgr.shape[:2]
#     x = max(0, x)
#     y = max(0, y)
#     w = min(w, w_img - x)
#     h = min(h, h_img - y)

#     # Apply padding to the bounding box
#     padding = 0.2
#     x_padded = max(0, x - int(w * padding))
#     y_padded = max(0, y - int(h * padding))
#     w_padded = min(w_img - x_padded, w + int(w * padding * 2))
#     h_padded = min(h_img - y_padded, h + int(h * padding * 2))

#     # Apply GrabCut algorithm
#     segmented_face = grabcut_algorithm(np_image_bgr, (x_padded, y_padded, w_padded, h_padded))

#     return segmented_face

# def grabcut_algorithm(original_image, bounding_box):
#     x, y, w, h = bounding_box

#     # Create a mask initialized with probable background
#     mask = np.zeros(original_image.shape[:2], np.uint8)
#     mask[:] = cv2.GC_PR_BGD

#     # Set the bounding box region as probable foreground
#     mask[y:y+h, x:x+w] = cv2.GC_PR_FGD

#     # Initialize the background and foreground models
#     background_model = np.zeros((1, 65), np.float64)
#     foreground_model = np.zeros((1, 65), np.float64)

#     # Run GrabCut algorithm
#     cv2.grabCut(original_image, mask, (x, y, w, h), background_model, foreground_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

#     # Extract the foreground mask
#     mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

#     # Refine the mask
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     # Create an alpha channel based on the mask
#     alpha_channel = mask.astype(np.uint8)

#     # Create a 4-channel image (RGBA) by adding the alpha channel to the original image
#     bgr_channels = cv2.split(original_image)
#     rgba_image = cv2.merge(bgr_channels + (alpha_channel,))  # Ensure it's a tuple


#     # Convert the RGBA image to PIL format
#     segmented_image_pil = Image.fromarray(cv2.cvtColor(rgba_image, cv2.COLOR_BGRA2RGBA))

#     return segmented_image_pil


# import cv2
# import numpy as np
# from PIL import Image
# from io import BytesIO
# import xml.etree.ElementTree as ET

# def decode_image(image_data):
#     try:
#         image = Image.open(BytesIO(image_data))
#         return image
#     except Exception as e:
#         raise ValueError(f"Invalid image data: {e}")

# def parse_xml_bounding_box(xml_data):
#     try:
#         root = ET.fromstring(xml_data)
#         object_element = root.find('object')
#         if object_element is None:
#             raise ValueError("Object element not found in the XML data")
        
#         bndbox = object_element.find('bndbox')
#         if bndbox is None:
#             raise ValueError("Bounding box element not found in the XML data")
        
#         xmin = bndbox.find('xmin')
#         ymin = bndbox.find('ymin')
#         xmax = bndbox.find('xmax')
#         ymax = bndbox.find('ymax')
        
#         if None in (xmin, ymin, xmax, ymax):
#             raise ValueError("Incomplete bounding box information in the XML data")
        
#         x = int(xmin.text)
#         y = int(ymin.text)
#         width = int(xmax.text) - x
#         height = int(ymax.text) - y
        
#         return x, y, width, height
#     except ET.ParseError as e:
#         raise ValueError(f"Error parsing XML data: {e}")
#     except Exception as e:
#         raise ValueError(f"Invalid XML data: {e}")

# def process_face_segmentation(image, bounding_box):
#     np_image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#     x, y, w, h = bounding_box

#     # Ensure the bounding box is within the image dimensions
#     h_img, w_img = np_image_bgr.shape[:2]
#     x = max(0, x)
#     y = max(0, y)
#     w = min(w, w_img - x)
#     h = min(h, h_img - y)

#     # Apply padding to the bounding box
#     padding = 0.2
#     x_padded = max(0, x - int(w * padding))
#     y_padded = max(0, y - int(h * padding))
#     w_padded = min(w_img - x_padded, w + int(w * padding * 2))
#     h_padded = min(h_img - y_padded, h + int(h * padding * 2))

#     # Apply GrabCut algorithm
#     segmented_face = grabcut_algorithm(np_image_bgr, (x_padded, y_padded, w_padded, h_padded))

#     return segmented_face

# def grabcut_algorithm(original_image, bounding_box):
#     x, y, w, h = bounding_box

#     # Create a mask initialized with probable background
#     mask = np.zeros(original_image.shape[:2], np.uint8)
#     mask[:] = cv2.GC_PR_BGD

#     # Set the bounding box region as probable foreground
#     mask[y:y+h, x:x+w] = cv2.GC_PR_FGD

#     # Initialize the background and foreground models
#     background_model = np.zeros((1, 65), np.float64)
#     foreground_model = np.zeros((1, 65), np.float64)

#     # Run GrabCut algorithm
#     cv2.grabCut(original_image, mask, (x, y, w, h), background_model, foreground_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

#     # Extract the foreground mask
#     mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

#     # Refine the mask
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     # Create an alpha channel based on the mask
#     alpha_channel = mask.astype(np.uint8)

#     # Create a 4-channel image (RGBA) by adding the alpha channel to the original image
#     bgr_channels = cv2.split(original_image)
#     rgba_image = cv2.merge(bgr_channels + (alpha_channel,))  # Ensure it's a tuple

#     # Convert the RGBA image to PIL format
#     segmented_image_pil = Image.fromarray(cv2.cvtColor(rgba_image, cv2.COLOR_BGRA2RGBA))

#     return segmented_image_pil


