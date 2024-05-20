from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import torch
from loguru import logger
import matplotlib.pyplot as plt


# Import your modules
from image_processing import PreprocessingPipeline
from model import HeadSegmentationModel
from segmentation_pipeline import HumanHeadSegmentationPipeline
from constants import HEAD_SEGMENTATION_MODEL_PATH
from visualization import VisualizationModule

import matplotlib
matplotlib.use('Agg')

import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/infiapp/Desktop/Hitesh_projects/caricature/face_cari/venv/lib/python3.12/site-packages/cv2/qt/plugins'

app = FastAPI()

MODEL_PATH = "model/head_segmentation.ckpt"

segmentation_pipeline = HumanHeadSegmentationPipeline(model_path=HEAD_SEGMENTATION_MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        image_bytes = await file.read()
        image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Predict the segmentation
        pred_segmap = segmentation_pipeline.predict(image)

        pred_segmap_np = np.array(pred_segmap)

        visualization_module = VisualizationModule()
        fig, _ = visualization_module.visualize_prediction(image, pred_segmap_np)

        # Save the visualization
        save_path = "segmentation_visualization.png"
        fig.savefig(save_path)
        plt.close(fig)  # Close the figure to release resources

        segmentation_shape = pred_segmap_np.shape

        return JSONResponse(content={"segmentation_map_shape": segmentation_shape, "save_path": save_path})
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error processing the image")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
