#code to detecting the rock paper scissor using yolov5 modle in real time
# Import required libraries
import cv2
import numpy as np

# Load the video and ONNX model
video = cv2.VideoCapture("-------path of your test video------")
model = cv2.dnn.readNetFromONNX("---path of best.onnxx file ------")

# Define class labels for the objects
class_labels = ['Paper', 'Rock', 'Scissors']

# Process each frame of the video
while True:
    # Read the frame from the video
    status, img = video.read()


    # Preprocess the image using the DNN function
    processed_img = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=(640,640), swapRB=True)
    model.setInput(processed_img)

    # Initialize lists for class boxes, scores, and ids
    class_box = []
    class_score = []
    class_id = []

    # Generate predictions from the model
    prediction = model.forward()[0]

    # Calculate the scaling factors for the bounding boxes
    width, height = img.shape[1], img.shape[0]
    x_scale = width / 640
    y_scale = height / 640

    # Iterate over each predicted object in the frame
    for i in range(prediction.shape[0]):
        row = prediction[i]
        max_score = row[4]

        # Check if the object is above the confidence threshold
        if max_score > 0.5:
            # Get the class with the highest score
            classes = row[5:]
            score_index = np.argmax(classes)

            # Check if the score for the class is above the threshold
            if classes[score_index] > 0.5:
                # Get the bounding box coordinates and scale them to the original image size
                cent_x, cent_y, old_w, old_h = row[:4]
                corn_x = int((cent_x - old_w/2) * x_scale)
                corn_y = int((cent_y - old_h/2) * y_scale)
                new_w = int(old_w * x_scale)
                new_h = int(old_h * y_scale)

                # Add the bounding box, score, and class id to their respective lists
                class_box.append([corn_x, corn_y, new_w, new_h])
                class_score.append(max_score)
                class_id.append(score_index)

    # Perform non-maximum suppression to remove duplicate bounding boxes
    objects = cv2.dnn.NMSBoxes(class_box, class_score, 0.5, 0.5)

    # Iterate over each remaining bounding box and draw it on the image
    for i in objects:
        x, y, w, h = class_box[i]
        label = class_labels[class_id[i]]
        score = class_score[i]
        text = f"{label} {score:0.2f}"

        # Assign a different color to each class and draw the bounding box and label text
        if label == 'Paper':
            color = (78, 158, 237) # shade of blue
        elif label == 'Rock':
            color = (213, 125, 240) # shade of purple
        else:
            color = (242, 128, 204) # shade of pink

        cv2.putText(img, text, (int(x), int(y-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)

    # Display the Video
    cv2.imshow('video',img)
    
    #press 'Q' to exit
    if cv2.waitKey(3)==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
