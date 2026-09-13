# AI Blind Assistant

An AI-based assistant designed to help visually impaired users understand and navigate their surroundings through real-time computer vision and voice interaction.

## Features

* Real-time object detection using YOLOv8
* Multi-object tracking using ByteTrack
* Approximate distance estimation using MiDaS monocular depth
* Scene understanding and room inference
* Object memory and semantic scene representation
* Risk assessment and navigation alerts
* Voice-based interaction
* LLM integration for scene-based questions and responses

## Technologies

* Python
* Flask
* YOLOv8
* ByteTrack
* MiDaS
* OpenCV
* Ollama
* Computer Vision
* Speech Recognition

## How It Works

The system captures the user's surroundings through a camera, detects and tracks objects, estimates their approximate distance, and builds an understanding of the scene.

The detected information is combined with object memory and risk assessment to provide useful voice-based alerts and answers.

## Project Structure

The project contains modules for:

* Object detection and tracking
* Depth and distance estimation
* Scene understanding
* Navigation and risk assessment
* Voice interaction
* Memory management
* Web interface

## Future Improvements

* OCR for reading text
* Arabic voice interaction
* Face recognition with user consent
* Sensor fusion
* Integration with smart glasses
