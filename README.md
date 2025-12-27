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

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/664f53ae-d5f6-4279-8b11-1461649769be" width="500"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/5ff70d16-1db0-4283-87cd-42b2b8e0d89c" width="500"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/8868ba3f-1ee7-4ea5-a3dc-78b623b57887" width="500"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/b48c0d4e-7f9c-4df9-a582-62491a1a1efe" width="500"/>
    </td>
  </tr>
</table>

