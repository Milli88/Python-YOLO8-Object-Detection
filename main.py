import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (or choose another)
model = YOLO('yolov8n.pt')

# Open the default camera (usually index 0)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If frame is read correctly, proceed with detection
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame)

    # Visualize the results on the frame (same as before)
    for result in results:
        for box in result.boxes:
            class_name = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0]

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {confidence:.2f}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame with detected objects
    cv2.imshow('Camera Object Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
