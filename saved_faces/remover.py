from PIL import Image, ImageOps
from rembg import new_session
import numpy as np
import cv2

class BackgroundRemover:
    def __init__(self, model_name="unetp"):
        self.session = new_session(model_name)
    
    def read_image(self, path):
        image = Image.open(path)
        return image
    
    def putalpha_cutout(self, img, mask):
        img.putalpha(mask)
        return img

    def get_concat_v(self, img1, img2):
        dst = Image.new("RGBA", (img1.width, img1.height + img2.height))
        dst.paste(img1, (0, 0))
        dst.paste(img2, (0, img1.height))
        return dst

    def get_concat_v_multi(self, imgs):
        pivot = imgs.pop(0)
        for im in imgs:
            pivot = self.get_concat_v(pivot, im)
        return pivot
    
    def apply_background(self, img, background, w, h):
        colored_image = Image.new("RGBA", (w, h), background)
        colored_image.paste(img, mask=img)
        return colored_image
    
    def process_image(self, img_path, background_path=None, default_background=(255, 255, 255), output_path=""):
        input_img = self.read_image(img_path)
        width, height = input_img.size
        input_img = ImageOps.exif_transpose(input_img)
        masks = self.session.predict(input_img)
        lst_imgs = []
        for mask in masks:
            cutout = self.putalpha_cutout(input_img, mask)
            lst_imgs.append(cutout)
        cutout = input_img
        if lst_imgs:
            cutout = self.get_concat_v_multi(lst_imgs)
        
        if background_path:
            background_img = self.read_image(background_path)
        else:
            background_img = default_background

        result = self.apply_background(cutout, background_img, width, height)

        if result.mode == "RGBA":
            result = result.convert("RGB")

        if output_path:
            result.save(output_path)
        return result


    def remove_face(image):
        # Load pre-trained face detection model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert image to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Create a mask for the face region
        mask = np.ones_like(gray) * 255
        for (x, y, w, h) in faces:
            cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 0), -1)  # Fill face region with black
        
        # Apply the mask to remove the face
        result = cv2.bitwise_and(image, image, mask=mask)

        return Image.fromarray(result)