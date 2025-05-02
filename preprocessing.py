import cv2

def preprocess_image(img_path):
    """
    Preprocess fingerprint image with Gaussian blur and Otsu's thresholding.
    """
    img = cv2.imread(img_path, 0)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh