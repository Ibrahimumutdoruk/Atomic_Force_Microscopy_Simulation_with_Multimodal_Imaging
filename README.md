# Atomic Force Microscopy Simulation with Multimodal Imaging

This repository presents a Python-based Atomic Force Microscopy (AFM) simulator with an interactive graphical user interface (GUI). The simulator is designed for educational and fundamental research purposes, enabling users to explore AFM imaging physics, tip–sample interactions, and multimodal surface characterization without the cost of real AFM hardware.

## Features

- Physics-based AFM simulation with Lennard Jones contact models  
- Configurable tip geometry (parabolic, conical, spherical) and cantilever parameters  
- Real-time simulation via a Tkinter-based GUI  
- Surface generation with multiple material types and roughness profiles  
- Multimodal AFM techniques:
  - Conductive AFM (C-AFM)
  - Kelvin Probe Force Microscopy (KPFM)
  - Scanning Thermal Microscopy (SThM)
- Noise and environmental effects (vibration, acoustic noise, humidity)
- 2D, 3D, spectral, and statistical visualization of AFM results  
## Graphical User Interface (GUI)
The GUI allows users to:
- Select AFM operating mode (contact, tapping, non-contact)
- Define surface type and scan size
- Adjust cantilever and tip parameters
- Enable multimodal imaging channels
- Visualize AFM topography, force–distance curves, PSD analysis, and 3D surfaces

All simulations are updated interactively.

## Requirements

- Python **3.9 or higher**
- Required Python packages:
  - numpy
  - scipy
  - matplotlib
  - tkinter (included with standard Python installations)
