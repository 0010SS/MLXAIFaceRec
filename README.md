# MLXAIFaceRec
Face Recognition using OpenCV Python. Resource for SCIE MLXAI ECA.

Welcome to the course material of MLXAI. In this repository, you can see the how to implement face recognition using OpenCV and Python.

To download OpenCV, please type in the following command in your command line (Terminal or CMD):

```
pip3 install opencv-python opencv-contrib-python
```

The `Model` file contains the files for face recognition pipeline.

- `ReadImageData.py` allows automatic capture of user's face and stores the faces as training example in `ImageDataset` folder.
- then, run `Trainer.py` to train the model
- `Recognizer.py` opens a recognizer for the face that has been stored using `ReadImageData.py`

**Cautious:** When using the recognizer, please modif the list `names` to the names of the faces you have recorded, following the order of id you've inputted in recording the faces.
