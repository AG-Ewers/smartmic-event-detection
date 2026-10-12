# SmartMic – event-driven detection of dividing cells on a UC2-based microscope

Code accompanying the Bachelor's thesis
**"Development of an AI-assisted microscopy workflow for automated event recognition"**
(Jakob Mastrandrea, Freie Universität Berlin, Ewers lab, 2026).

The software controls a UC2-based fluorescence microscope (FYSETC E4 board with FluidNC,
Allied Vision Alvium camera, 638 nm laser). At each position, the current camera image can be
analysed with a YOLOv8 model. If a dividing cell is detected, a z-stack and an exposure series
are recorded automatically.

## Repository structure

| Folder | Content |
|---|---|
| `software/` | PySide6 control software (`main.py` and one module per hardware component) |
| `firmware/` | FluidNC configuration for the FYSETC E4 board |
| `training/` | Jupyter notebook used to train the YOLOv8 models (Google Colab) |
| `model/` | Trained model weights used in the thesis (`best_v7_32.pt`) |
| `analysis/` | Scripts for the evaluation table and the result figures |

## Installation

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

The camera is accessed through VmbPy, which requires the Allied Vision **Vimba X SDK**.
Install the SDK first and then VmbPy as described by Allied Vision.

## Usage

1. Connect the FYSETC E4 board via USB and set the serial port in `software/settings.py` (`GRBL_PORT`).
2. Check the path to the model weights in `software/event_detector.py`.
3. Start the software:

```bash
cd software
python main.py
```

4. Connect the motion controller and the camera in the interface, move to a position, and press
   **Analysieren** (analyze) to run the detection and, if a dividing cell is found, the acquisition series.

## Notes

- Code comments and interface labels are in German.
- The autofocus and the pixel-to-stage calibration are implemented but were not validated
  and were not used in the thesis.
- The version used in the thesis is tagged as `v1.0-thesis`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
<!-- License to be chosen together with the Ewers lab, e.g. MIT -->
