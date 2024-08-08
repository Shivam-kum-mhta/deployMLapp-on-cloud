# Import libraries
from ultralytics import YOLO
import cv2
import numpy as np
import logging
# Load the model
model = YOLO('yolov10n.pt')

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse
from typing import List
import shutil
app = FastAPI()

def detection (input_video_path):
  count = 0

  # Open the video
  cap = cv2.VideoCapture(input_video_path)

  # Get video properties, to be used to process
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  fps = cap.get(cv2.CAP_PROP_FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  output_video_path = input_video_path.split('.')[0] + '-out.mp4'

  # Number of skips
  num_skips = frames // 200
  print(f"Frames = {frames}, Number of skips: {num_skips}")

  # Define the codec and create VideoWriter object - to write output video
  out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

  # Until there is a frame left, run the loop - skip num_skip frames
  while cap.isOpened():
    ret, frame = cap.read()
    # If the frame is not there, break
    for i in range(num_skips):
      if not ret:
          break
      ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model on the frame selected
    results = model(frame)
    count += 1

    # For all the result boxes, plot them into a frame
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    for i in range(num_skips + 1):
      out.write(annotated_frame)

  # Close everything
  cap.release()
  out.release()
  cv2.destroyAllWindows()
  print(f"{count} frames processed.")
  return output_video_path


# Directory to save uploaded files temporarily
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Create routers
image_router = APIRouter()
video_router = APIRouter()

@image_router.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Verify file type
    if file.content_type not in ["image/jpeg", "image/png", "image/gif"]:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # You can save the file here or process it directly
    # For demonstration, we just return the file name
    return {"filename": file.filename}

@video_router.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    # Verify file type
    if file.content_type not in ["video/mp4", "video/mpeg", "video/avi"]:
        raise HTTPException(status_code=400, detail="Invalid video format")

    # You can save the file here or process it directly
        # Save the file to the upload directory
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print("file_path ============================== " , file_path)
    # You can process the file here if needed
    output_file_path=detection(file_path)
    # Return the file as a downloadable response
    return FileResponse(output_file_path, media_type=file.content_type, filename=file.filename)


    # For demonstration, we just return the file name
    detection(file)
    return {"filename": file.filename}

# Register routers with the FastAPI app
app.include_router(image_router, prefix="/image")
app.include_router(video_router, prefix="/video")

# Root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Welcome to the image and video upload API"}

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=8001)

