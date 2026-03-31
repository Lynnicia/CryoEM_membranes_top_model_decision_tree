# my_module/file.py

def main():
    print("Running file.py as module")

if __name__ == "__main__":
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Loading YOLOv11 and YOLO26...")
from ultralytics import YOLO

#### Monkey Patch when saving in NOLOSS #############
import ultralytics.data.loaders as loaders

# Step 1: Save the original YOLO normalization function
_orig = loaders.LoadTensor._single_check

# Step 2: Define a patch that skips extra normalization
def _patched(im, stride=32):
    return _orig(im, stride=stride)  # Do shape check only, no undo/redo of normalization

# Step 3: Apply the patch
loaders.LoadTensor._single_check = staticmethod(_patched)
print("✓ YOLOv11 and YOLO26 model architectures loaded")