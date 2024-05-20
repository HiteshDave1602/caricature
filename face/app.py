# from fastapi import FastAPI, File, UploadFile
# from PIL import Image, ImageDraw
# import torch
# import torchvision.transforms as T
# from io import BytesIO
# from fine_tune import get_model
# import base64

# app = FastAPI()

# model = get_model(num_classes=2)
# model.load_state_dict(torch.load("head_segmentation_model.pth"))
# model.eval()

# transform = T.Compose([T.ToTensor()])

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     contents = await file.read()
#     image = Image.open(BytesIO(contents)).convert("RGB")
#     image_tensor = transform(image).unsqueeze(0)

#     with torch.no_grad():
#         prediction = model(image_tensor)
    
#     boxes = prediction[0]["boxes"].cpu().numpy()
    
#     # Generate masks for each bounding box
#     masks = []
#     for box in boxes:
#         mask = generate_mask(image.size, box)
#         masks.append(mask)
    
#     # Save the masks locally
#     save_masks_locally(masks)
    
#     # Convert masks to base64-encoded strings
#     encoded_masks = [encode_mask(mask) for mask in masks]
    
#     return {"masks": encoded_masks}

# def generate_mask(image_size, box):
#     mask = Image.new("L", image_size, 0)
#     draw = ImageDraw.Draw(mask)
#     draw.rectangle(box.tolist(), fill=255)
#     return mask

# def save_masks_locally(masks):
#     for i, mask in enumerate(masks):
#         mask.save(f"head_mask_{i}.png")

# def encode_mask(mask):
#     buffered = BytesIO()
#     mask.save(buffered, format="PNG")
#     encoded_string = base64.b64encode(buffered.getvalue()).decode()
#     return encoded_string

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# import base64
# from io import BytesIO
# from fastapi import FastAPI, File, UploadFile
# from PIL import Image
# import torch
# import torchvision.transforms as T
# import numpy as np

# from torchvision.models.segmentation import deeplabv3_resnet50

# app = FastAPI()

# # Load the pre-trained mask prediction model
# model = deeplabv3_resnet50(pretrained=False, num_classes=2)
# model = torch.load("head_segmentation_model.pth",map_location=torch.device('cpu'))
# model.eval()

# # Define image transformations
# transform = T.Compose([
#     T.Resize((256, 256)),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     contents = await file.read()
#     image = Image.open(BytesIO(contents)).convert("RGB")
    
#     # Preprocess the image
#     input_tensor = transform(image).unsqueeze(0)
    
#     # Perform inference
#     with torch.no_grad():
#         output = model(input_tensor)
    
#     # Convert output to segmentation mask
#     mask = torch.argmax(output, dim=1).squeeze().numpy()
    
#     # Perform any post-processing (optional)
#     # For example, you can apply thresholding or morphological operations
    
#     # Apply the mask to the original image
#     masked_image = apply_mask(image, mask)
    
#     # Convert the masked image to bytes
#     buffered = BytesIO()
#     masked_image.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
#     # Return the masked image
#     return {"masked_image": img_str}

# def apply_mask(image, mask):
#     # Create a mask image from the segmentation mask
#     mask_image = Image.fromarray((mask * 255).astype(np.uint8))
#     # Apply the mask to the original image
#     masked_image = Image.composite(image, Image.new('RGB', image.size, (0, 0, 0)), mask_image)
#     return masked_image

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



import torch,base64
from torchvision.models.segmentation import deeplabv3_resnet50
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torchvision.transforms as T
import numpy as np
import io

app = FastAPI()

# Load the pre-trained mask prediction model
model = deeplabv3_resnet50(pretrained=False, num_classes=2)  # Assuming you're using a DeepLabV3 model
model.load_state_dict(torch.load("head_segmentation_model.pth", map_location=torch.device('cpu')))  # Load the model state dict
model.eval()

# Define image transformations
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0)
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)['out']  # Access the output of the model
    
    # Convert output to segmentation mask
    mask = torch.argmax(output, dim=1).squeeze().numpy()
    
    # Apply the mask to the original image
    masked_image = apply_mask(image, mask)
    
    # Convert the masked image to bytes
    buffered = io.BytesIO()
    masked_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Return the masked image
    return {"masked_image": img_str}

def apply_mask(image, mask):
    # Create a mask image from the segmentation mask
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    # Apply the mask to the original image
    masked_image = Image.composite(image, Image.new('RGB', image.size, (0, 0, 0)), mask_image)
    return masked_image

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
