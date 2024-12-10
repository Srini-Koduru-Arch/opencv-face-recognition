FROM python:3.10

# Set the working directory in the container

WORKDIR /opencv-fd

COPY . /opencv-fd

RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit will run on
EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "opencv_fd_app.py", "--server.port=8051", "--server.headless", "true"]