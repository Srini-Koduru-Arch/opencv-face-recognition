FROM python:3.10

EXPOSE 8080
WORKDIR /opencv-face-recognition

COPY . ./

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "opencv_fd_app.py", "--server.port=8080", "--server.address=0.0.0.0"]