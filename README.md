# StructuredLightCalibration-Python

This project including:

- Gray code & Phase shifting pattern generation
- Data collection with camera and projector
    - Virtual
    - Real one
- Decoding with given gray code + phase shifting to get accurate correspondence
- Calibration process for sturctured light systems

## Requirements:

- Basic requirements: Including numpy, opencv, etc. (TODO)
- ext/pointerlib.whl: A package written by myself. Including some useful functions.
- ext/PySpin: The driver for my camera.

## Usage:

### 0. Install all the requirements

### 1. Generate pattern

```python
python generate_pattern.py
```

### 2. Collect data with virtual or real sensors

```python
python collect_data.py
```

### 3. Decode to get the coordinate mats

```python
python decode_scene_grayphase.py
```

### 4. Calibration

```python
python calibrate.py
```
