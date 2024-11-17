
# Face Recognition Attendance System

This is a Flask-based web application that uses face recognition to mark attendance. The system detects faces using OpenCV's Haar Cascade classifier and recognizes them using a K-Nearest Neighbors (KNN) classifier.

## Features

- **Face Detection**: Detects faces in real-time using OpenCV.
- **Face Recognition**: Identifies faces using a pre-trained KNN model.
- **Attendance Logging**: Logs attendance with the name, roll number, and time.
- **User Management**: Allows adding new users with multiple face images for training.

## Requirements

- Python 3.x
- Flask
- OpenCV
- NumPy
- Scikit-learn
- Pandas
- Joblib

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/face-recognition-attendance.git
   cd face-recognition-attendance
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare directories**:
   Ensure the following directories are created:
   - `Attendance`
   - `static`
   - `static/faces`

4. **Download Haar Cascade classifier**:
   Download the `haarcascade_frontalface_default.xml` file and place it in the root directory of the project. You can download it from [here](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml).

## Usage

1. **Run the Flask application**:
   ```bash
   python app.py
   ```

2. **Access the application**:
   Open your web browser and go to `http://127.0.0.1:5000/`.

3. **Add New User**:
   - Go to the `/add` route.
   - Enter the username and roll number.
   - Capture face images by following the on-screen instructions.

4. **Start Attendance**:
   - Go to the `/start` route.
   - The system will start the camera and recognize faces in real-time.
   - Attendance will be logged in the `Attendance` directory with the current date.

## File Structure

- `app.py`: The main Flask application file.
- `background.png`: Background image for the attendance display.
- `haarcascade_frontalface_default.xml`: Haar Cascade classifier for face detection.
- `requirements.txt`: List of required Python packages.
- `static/`: Contains the trained model and face images.
  - `faces/`: Directory to store user face images.
- `Attendance/`: Directory to store attendance logs.

## Functions

- `totalreg()`: Returns the total number of registered users.
- `extract_faces(img)`: Detects faces in the given image.
- `identify_face(facearray)`: Identifies the face using the pre-trained model.
- `train_model()`: Trains the KNN model with the registered user faces.
- `extract_attendance()`: Extracts attendance data from the CSV file.
- `add_attendance(name)`: Adds the user's attendance to the CSV file.
- `getallusers()`: Retrieves all registered users.

## Routes

- `/`: Home route displaying the attendance log.
- `/start`: Starts the real-time face recognition for attendance.
- `/add`: Adds a new user with face images for training.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenCV](https://opencv.org/) for face detection and image processing.
- [Flask](https://flask.palletsprojects.com/) for creating the web application.
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms.
