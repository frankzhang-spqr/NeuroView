# NeuroView MRI Technical Specification
## Automated Volumetric MRI Analysis and Brain Tumor Segmentation Suite

NeuroView MRI is a high-performance desktop application designed for the clinical visualization of NIfTI-format MRI data and the automated identification of pathological regions using deep learning. This document provides a comprehensive technical overview of the system architecture, codebase, and deployment pipeline.

---

## 1. System Architecture

NeuroView utilizes a decoupled architecture comprising a high-fidelity Graphical User Interface (GUI) and a specialized Computational Backend.

### 1.1 Renderer Process (Electron/JavaScript)
The frontend is built on the Electron framework, utilizing the Chromium rendering engine. It is responsible for:
- **Volumetric Visualization**: Implementing a Three.js-based WebGL environment for real-time 3D raymarching.
- **User Interface State**: Managing a complex state machine for multi-modal synchronization and analysis workflows.
- **Asynchronous Communication**: Interfacing with the backend via a low-latency REST API and internal IPC bridges.

### 1.2 Computational Service (Python/FastAPI)
The backend is a high-performance Python service bundled as a native Windows executable. Its responsibilities include:
- **NIfTI Processing**: utilizing the `nibabel` library to parse and manipulate complex medical imaging volumes.
- **Numerical Normalization**: Implementing global volumetric intensity normalization to ensure clinical contrast across T1c, T1n, T2f, and T2w modalities.
- **AI Inference Engine**: Running PyTorch-based UNet models for pixel-level tumor segmentation.

---

## 2. Codebase Specification

### 2.1 Core Application Logic (main.js)
The entry point for the Electron main process. This file manages the application lifecycle and secure system integration.
- **Process Management**: Handles the instantiation and termination of the Python backend service, including PID tracking to prevent zombie processes.
- **Native Interfacing**: Implements the native Windows menu system and high-level file system dialogs.
- **Security Protocols**: Configures the context isolation and preload scripts to ensure the UI has no direct access to system-level Node.js APIs.

### 2.2 Backend Service Implementation (app/main.py)
The primary service layer for the application.
- **Endpoint Specification**: Defines RESTful routes for fetching normalized 32-bit floating-point volume data and segmented tumor masks.
- **Resource Management**: Implements caching for NIfTI volumes to minimize I/O overhead during slice navigation.
- **AI Integration**: Orchestrates the loading of weights from `.pth` files and the execution of the segmentation pipeline on the detected hardware (CPU or CUDA).

### 2.3 UI & Rendering Engine (app/static/dashboard.js)
The primary logic controller for the renderer process.
- **WebGL Environment**: Initializes the Three.js scene, including the camera, lighting, and orbit controls.
- **Raymarching Shaders**: Contains the GLSL fragment shaders used to perform volumetric rendering of the MRI data directly from binary buffers.
- **Slice Synchronization**: Implements the logic that maps 2D coordinates from the Axial, Coronal, and Sagittal views to the 3D space of the volume renderer.

### 2.4 Preload Integration (preload.js)
A secure bridge that exposes a limited set of IPC capabilities to the frontend, allowing the dashboard to communicate with the Electron main process without compromising security.

### 2.5 Distribution Pipeline (build_dist.ps1)
A PowerShell-based build automation script.
- **Stage 1 (Cleanup)**: Ensures a deterministic build environment by purging all temporary artifacts.
- **Stage 2 (Backend Compilation)**: Executes PyInstaller with specific hooks to bundle the PyTorch engine, including aggressive pruning of training-only libraries to optimize disk footprint.
- **Stage 3 (Resource Preparation)**: Manually constructs the application bundle folder, ensuring all static assets and AI weights are in the correct relative paths.
- **Stage 4 (Multi-Part Packaging)**: Compresses the application into a high-speed `.7z` archive and prepares the `Setup.bat` installer.

---

## 3. Data Processing & AI Methodology

### 3.1 Volumetric Normalization
To ensure the AI models receive consistent data, the backend performs global intensity normalization on every loaded volume. This process transforms the raw Hounsfield or intensity units into a standardized 0-1 floating-point range, preserving the relative contrast necessary for accurate pathological identification.

### 3.2 UNet Segmentation
The AI engine utilizes a deep Convolutional Neural Network based on the UNet architecture.
- **Multi-Modal Input**: The model accepts 4 input channels (T1c, T1n, T2f, T2w) to provide a holistic view of the brain tissue.
- **Inference Pipeline**: The system performs automated brain-masking, cropping, and 3D-to-2D projection before executing the segmentation model.
- **Post-Processing**: The resulting probability masks are thresholded and converted into color-coded overlays for the final user display.

---

## 4. Hardware and Software Requirements

### 4.1 System Requirements
- **Operating System**: Windows 10 or 11 (64-bit).
- **Processor**: Intel Core i5 (8th Gen) or AMD Ryzen 5 minimum.
- **Memory**: 8GB RAM (16GB highly recommended for 3D volume rendering).
- **Graphics**: Integrated graphics supported; NVIDIA GPU with CUDA support recommended for accelerated AI inference.

### 4.2 Software Dependencies
- **Runtime**: Electron 41.2.1, Python 3.9 (bundled).
- **Medical Libraries**: Nibabel, NumPy, Scikit-Image.
- **AI Framework**: PyTorch 2.0+.

---

## 5. Deployment and Installation

NeuroView is distributed as a multi-part installation package to ensure stability across all Windows environments.
1. **Extraction**: The user extracts the distribution ZIP folder.
2. **Setup**: The user executes `Setup.bat`, which runs a PowerShell installation script.
3. **Integration**: The script installs the app to the local AppData directory, creates Start Menu shortcuts, and registers the application for searchability.

---

© 2026 NeuroView Team. All Rights Reserved.
Technical manual version 1.0.0.