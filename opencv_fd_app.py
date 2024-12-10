import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
from io import BytesIO
import base64


st.title("Face Detection - Based on OpenCV Deep Learning")
img_file_buffer = st.file_uploader("Upload an Image (jpg, jpeg and png)", type=['jpg', 'jpeg', 'png'])

threshold = 0.7

#Detect faces using OpenCV
def detect_face_opencv_dnn(net, frame):
    #get the image blob prepared with preprocessing
    blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    #set blob as input to the model
    net.setInput(blob)
    #get detections
    detections = net.forward()
    return detections

#Annote the image with bounding box
def process_detections(frame, detections, conf_threshold=threshold):
    bboxes = []
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    # Loop over all detections and draw bounding boxes around each face.
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            bboxes.append([x1, y1, x2, y2])
            bb_line_thickness = max(1, int(round(frame_h/200)))
            # Draw bounding boxes around detected faces.
            cv.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), bb_line_thickness, cv.LINE_8)
    return frame, bboxes

#Function to load the DNN model
@st.cache_resource
def load_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv.dnn.readNetFromCaffe(configFile, modelFile)
    return net

# Function to generate a download link for output file.
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

net = load_model()

if img_file_buffer is not None:
    #Read the file and convert it to opencv Image.
    raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    #load image in BGR channel order
    image = cv.imdecode(raw_bytes, cv.IMREAD_COLOR)

    # Or use PIL Image (which uses an RGB channel order)
    # image = np.array(Image.open(img_file_buffer))

    # Create placeholders to display input and output images.
    placeholders = st.columns(2)
    # Display Input image in the first placeholder.
    placeholders[0].image(image, channels='BGR')
    placeholders[0].text("Input Image")

    # Create a Slider and get the threshold from the slider.
    conf_threshold = st.slider("SET Confidence Threshold", min_value=0.0, max_value=1.0, step=.01, value=threshold)

    # Call the face detection model to detect faces in the image.
    detections = detect_face_opencv_dnn(net, image)

    # Process the detections based on the current confidence threshold.
    out_image, _ = process_detections(image, detections, conf_threshold=conf_threshold)

    # Display Detected faces.
    placeholders[1].image(out_image, channels='BGR')
    placeholders[1].text("Output Image")

    # Convert opencv image to PIL.
    out_image = Image.fromarray(out_image[:, :, ::-1])
    # Create a link for downloading the output file.
    st.markdown(get_image_download_link(out_image, "face_output.jpg", 'Download Output Image'), unsafe_allow_html=True)