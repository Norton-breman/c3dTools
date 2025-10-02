# C3DTools

A collection of tools to analyse data from *C3D files opened with [pupyC3D](https://pypi.org/project/pupyC3D/)

## Features

- **Extract force plate data (Needs verification)**

## Usage

```python
from pupyC3D import C3DFile
import matplotlib.pyplot as plt
from c3dTools.force_plate import extract_force_plates

# Load a C3D file
c3d = C3DFile('example.c3d')
fps = extract_force_plates(c3d)
for i, f in enumerate(fps):
    print(f.description)
    print(f.corners)
    plt.figure(i)
    force = f.get_force()
    plt.plot(force)
    plt.legend(['x', 'y', 'z'])
    plt.title(f.description)
plt.show()
