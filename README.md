# Face Detection and Recognition System

A local system for face detection and recognition based on stored reference patterns. Designed to test and improve algorithm robustness under various lighting conditions, this project enables adding and managing face templates, performing real-time recognition, and analyzing system performance in realistic scenarios.

---

## 📌 About the Project

The goal of this project was to create a system that:

- Detects and recognizes faces from a camera feed or video files
- Allows users to add and manage their own reference face patterns
- Evaluates recognition performance under different lighting conditions

Key design principles:
- Fully local processing (no data sent to external servers)
- Cross-platform support (Windows, Linux)
- Easy installation using Docker
- Intuitive graphical user interface (GUI) built with Tkinter

Tests demonstrated ~90% recognition accuracy in optimal lighting. Analysis showed significant drops in accuracy under extreme lighting variations, motivating future improvements in preprocessing.

---

## 🎯 Key Features

✅ Add new face templates from camera or video  
✅ Automatic best-frame selection for template creation  
✅ Real-time face recognition  
✅ GUI for easy interaction  
✅ CLI mode for advanced usage  
✅ Lighting-condition testing and analysis  
✅ Fully local operation without external servers  
✅ Dockerized deployment

---

## 🧪 Research Focus

This project includes an experimental component analyzing how lighting conditions affect recognition accuracy.

Test scenarios:

- **Ideal lighting**
- **Low light (brightness 13–27)**
- **Overexposed / very bright (brightness 106–120)**

**Results:**

- ~93% accuracy in normal conditions
- ~60–68% accuracy in low-light conditions
- ~85% accuracy in moderate overexposure

**Conclusions:**

- System handles overexposure better than low-light scenarios
- Further improvements needed for preprocessing under poor lighting

---

## 🛠️ Built With

- **Python 3.10**
- **OpenCV** – image capture and processing
- **PyTorch** – FaceSDK backend
- **FaceX-Zoo (FaceSDK)** – face detection and recognition
- **Tkinter** – graphical user interface
- **Docker / Docker Compose** – easy deployment
- **NumPy, Pillow, scikit-image, matplotlib** – supporting libraries
