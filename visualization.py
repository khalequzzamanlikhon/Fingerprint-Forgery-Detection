import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import skimage.morphology

def show_minutiae_results(skel, term_label, bif_label):
    """
    Visualize minutiae extraction results.
    """
    (rows, cols) = skel.shape
    disp_img = np.zeros((rows, cols, 3), np.uint8)
    disp_img[:, :, 0] = skel
    disp_img[:, :, 1] = skel
    disp_img[:, :, 2] = skel
    
    rp = skimage.measure.regionprops(bif_label)
    for i in rp:
        (row, col) = np.int16(np.round(i['Centroid']))
        (rr, cc) = skimage.draw.circle_perimeter(row, col, 1)
        disp_img[rr, cc] = (255, 0, 0)  # Red for bifurcations
    
    rp = skimage.measure.regionprops(term_label)
    for i in rp:
        (row, col) = np.int16(np.round(i['Centroid']))
        (rr, cc) = skimage.draw.circle_perimeter(row, col, 1)
        disp_img[rr, cc] = (0, 0, 255)  # Blue for terminations
    
    plt.figure(figsize=(6, 6))
    plt.title("Minutiae Extraction (Red: Bifurcations, Blue: Terminations)")
    plt.imshow(disp_img)
    plt.axis('off')
    plt.show()

def test_single_image(model, image_path, preprocess_image, get_terminations_bifurcations, conf_threshold=0.5):
    """
    Test YOLO model on a single image with minutiae extraction.
    """
    results = model.predict(image_path, conf=conf_threshold)
    label = "UNCERTAIN (No detection)"
    cls = None
    conf = None
    
    if len(results[0].boxes) > 0:
        cls = int(results[0].boxes.cls[0])
        conf = float(results[0].boxes.conf[0])
        label = f"{'REAL' if cls == 0 else 'FAKE'} ({conf:.2f})"
    
    img = preprocess_image(image_path)
    skel = skimage.morphology.skeletonize(img > 0)
    skel = np.uint8(skel) * 255
    mask = img * 255
    minutiae_term, minutiae_bif = get_terminations_bifurcations(skel, mask)
    term_label = skimage.measure.label(minutiae_term, connectivity=1)
    bif_label = skimage.measure.label(minutiae_bif, connectivity=1)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(results[0].plot()[..., ::-1])
    plt.title(label, fontsize=12)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    show_minutiae_results(skel, term_label, bif_label)
    plt.show()
    
    if cls is not None:
        print(f"Prediction: {'Real' if cls == 0 else 'Fake'}")
        print(f"Confidence: {conf:.4f}")
    else:
        print("No fingerprint detected with sufficient confidence")