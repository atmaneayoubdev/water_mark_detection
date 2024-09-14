from ultralytics import YOLO
import cv2
import torch


def process_image(img_path, model):
    # Load the image
    img = cv2.imread(img_path)

    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.amp.autocast(device_type=device_type):
         # Run inference
        results = model(img)  # Inference will automatically use the right device

    # Display and save each result
    for result in results:
        result.show()  # Show the output image with bounding boxes
        result.save()  # Save the results (with bounding boxes) in the current directory

        # Access detailed information about the predictions
        boxes = result.boxes.xyxy  # Bounding box coordinates (x1, y1, x2, y2)
        confs = result.boxes.conf  # Confidence scores
        labels = result.boxes.cls  # Class labels

        # Print results (if needed)
        for i in range(len(boxes)):
            print(f"Label: {labels[i]}, Confidence: {confs[i]}, Box: {boxes[i]}")

def main():
    # Load the model (preload only once)
    model = YOLO('models/best_3.pt')

    if torch.cuda.is_available():
        model.cuda()  # Move model to GPU if available

    # Paths to the images you want to test
    img_paths = ['images/test_3.webp']  # You can add more images to the list

    for img_path in img_paths:
        process_image(img_path, model)

if __name__ == "__main__":
    main()
