# 🚨 Accident Detection and Reporting System

A deep learning-based system that detects accidents from video footage and sends instant alerts with geolocation via SMS using Twilio. Built with OpenCV, VGG16, and geolocation APIs, this project aims to automate emergency responses to traffic accidents.

---

## 📌 Features

- ✅ **Accident detection using deep learning (VGG16 + Dense Layers)**
- 📹 **Video frame processing with OpenCV**
- 🌐 **Geolocation detection using IP-based coordinates**
- 📩 **Instant SMS alert via Twilio API**
- 📊 **Visual feedback using OpenCV frame overlay**

---

## 🎯 How It Works

1. **Training Phase**
   - Extracts frames from `Accidents.mp4`
   - Preprocesses and resizes images
   - Trains a neural network using pre-trained VGG16 for feature extraction

2. **Prediction Phase**
   - Reads test frames from `Accident-1.mp4`
   - Uses trained model to classify frames: `Accident` or `No Accident`
   - Displays real-time prediction on video playback

3. **Alerting System**
   - On detecting an accident, fetches user's location
   - Sends SMS to a predefined number using Twilio with the location address

---

## 📦 Tech Stack:

| Tool        | Purpose                          |
|-------------|----------------------------------|
| `Python`    | Main programming language        |
| `OpenCV`    | Video processing and display     |
| `VGG16`     | Deep learning feature extractor  |
| `Keras`     | Neural network building/training |
| `Twilio`    | SMS notification API             |
| `Geocoder`  | IP-based location detection      |
| `Geopy`     | Reverse geocoding                |
| `Pandas`    | CSV data handling                |
| `Matplotlib`| Visualization                    |

---

## 📸 Screenshots:
!(grafana/Dashboard.png)
---
