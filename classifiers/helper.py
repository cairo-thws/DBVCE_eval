from . import MadryClassifier, AlexNetClassifier, AlexNetRandomInitClassifier, UnetEncoderClassifier,ConvNeXtLClassifier, SIMCLRClassifier, SwinTFLClassifier

def get_classifiers():
    return [MadryClassifier(), AlexNetClassifier(), AlexNetRandomInitClassifier(), UnetEncoderClassifier(),ConvNeXtLClassifier(), SIMCLRClassifier(), SwinTFLClassifier() ]