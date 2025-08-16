from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow

# Load your trained model (replace path if needed)
model = YOLO("runs/detect/train5/weights/best.pt")

# Run inference on an image
results = model("/content/thermal.jpg")  # example image

# Show results
for r in results:
    im_array = r.plot()  # plot boxes on image
    cv2_imshow(im_array)  # display in colab