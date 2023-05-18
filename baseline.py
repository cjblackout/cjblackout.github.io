import numpy as np
import cv2
import pymeanshift as pms
from fastapi import FastAPI, UploadFile, File
import io
from fastapi.responses import StreamingResponse
import logging
import gc

THRESHOLD = 0.5
SPATIAL_RADIUS = 2
RANGE_RADIUS = 8
MIN_DENSITY = 150
PROCESSING_SHAPE = (256,256)

logging.basicConfig(filename='server.log', encoding='utf-8', level=logging.DEBUG)

app = FastAPI()

@app.post("/baseline_mean_shift_app")
async def baseline_mean_shift_app(image: UploadFile = File(...), processing_shape: tuple = (256,256),
                              spatial_radius: int = SPATIAL_RADIUS, range_radius: int = RANGE_RADIUS,
                              min_density: int = MIN_DENSITY):
    """
    Perform baseline mean shift segmentation on an uploaded image and return the resulting image with mask applied.

    Parameters:
    - image: UploadFile (required) - The image file to be processed.
    - processing_shape: tuple - The shape (width, height) to which the image should be resized for processing. Default: PROCESSING_SHAPE.
    - spatial_radius: int - The spatial radius parameter for mean shift. Default: SPATIAL_RADIUS.
    - range_radius: int - The range radius parameter for mean shift. Default: RANGE_RADIUS.
    - min_density: int - The minimum density parameter for mean shift. Default: MIN_DENSITY.

    Returns:
    - StreamingResponse: The processed image with mask applied as a streaming response in JPEG format.

    Raises:
    - Exception: If there is an error during image processing.

    Example usage:

    files = {'image': open(image_file_path, "rb")}
    response = requests.post(url, files=files)

    if response.status_code == 200:
    
        with open('path/to/image.jpg', "wb") as f:
            f.write(response.content)
        print(f"Segmented image saved successfully at: {'image.jpg'}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

    """
    
    try:
        # Read the uploaded image as bytes

        contents = image.file.read()
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        mask, _, _ = baseline_mean_shift(img, processing_shape=processing_shape, spatial_radius=spatial_radius, range_radius=range_radius, min_density=min_density)
        after_img = cv2.bitwise_and(img, img, mask=mask)

        # Convert the segmented image numpy array to bytes
        _, after_img_bytes = cv2.imencode(".jpg", after_img)

        #clear memory
        del image
        del contents
        del img_array
        del img
        del mask
        del after_img
        gc.collect()

        # Return the after image as a streaming response
        return StreamingResponse(io.BytesIO(after_img_bytes), media_type="image/jpeg")

    except Exception as e:
        logging.error(e)
        return {"error": str(e)}

def baseline_mean_shift(img, processing_shape=PROCESSING_SHAPE, spatial_radius=SPATIAL_RADIUS, range_radius=RANGE_RADIUS, min_density=MIN_DENSITY):
    """
    Applies baseline mean shift segmentation to an input image and generates a mask for the dominant region.

    Parameters:
        img (numpy.ndarray): The input image to be segmented.
        processing_shape (tuple, optional): The shape to resize the input image for processing. Defaults to PROCESSING_SHAPE.
        spatial_radius (int, optional): The spatial radius parameter for mean shift segmentation. Controls the size of the spatial neighborhood. Defaults to SPATIAL_RADIUS.
        range_radius (int, optional): The range radius parameter for mean shift segmentation. Controls the size of the color range neighborhood. Defaults to RANGE_RADIUS.
        min_density (int, optional): The minimum density parameter for mean shift segmentation. Controls the minimum number of pixels in a region. Defaults to MIN_DENSITY.

    Returns:
        mask (numpy.ndarray): The binary mask representing the dominant region (e.g., sky) in the original image shape.
        segmented_image (numpy.ndarray): The segmented image with labeled regions.
        labels_image (numpy.ndarray): The image with labels corresponding to each region.

    Raises:
        ValueError: If the input image is not a valid numpy.ndarray.
        Exception: If an error occurs during mean shift segmentation.

    Example usage:
        try:
            img = cv2.imread('image.jpg')
            mask, segmented_image, labels_image = baseline_mean_shift(img)
        except ValueError as ve:
            print(f"Invalid input image: {ve}")
        except Exception as e:
            print(f"Error occurred during mean shift segmentation: {e}")

    """

    try:
        # Check if the input image is a valid numpy.ndarray
        if not isinstance(img, np.ndarray):
            logging.error("Input image must be a numpy.ndarray")
            raise ValueError("Input image must be a numpy.ndarray")
        
        # Save the original shape of the image, resize it, and segment the image
        original_shape = img.shape
        try:
            img = cv2.resize(img, (256,256))
        except Exception as e:
            logging.error(e)
            raise Exception("Error occurred during image resizing.") from e

        #perform mean shift segmentation
        try:
            (segmented_image, labels_image, number_regions) = pms.segment(img, spatial_radius=spatial_radius, range_radius=range_radius, min_density=min_density)
        except Exception as e:
            logging.error(e)
            raise Exception("Error occurred in pymeanshift function.") from e

        # Take the upper half of labels_image and determine the most dominant label in the upper half
        upper_labels = labels_image[0:labels_image.shape[0]//2, 0:labels_image.shape[1]]
        unique, counts = np.unique(upper_labels, return_counts=True)
        dominant_label = unique[np.argmax(counts)]

        # Create a mask from labels_image where the dominant_label represents the dominant region (e.g., sky) and resize it to the original shape
        mask = np.zeros(labels_image.shape, dtype=np.uint8)
        mask[labels_image == dominant_label] = 1
        mask = cv2.resize(mask, (original_shape[1], original_shape[0]))

        return mask, segmented_image, labels_image

    except ValueError as ve:
        raise ve
    except Exception as e:
        logging.error(e)
        raise Exception("Error occurred during sky detection.") from e