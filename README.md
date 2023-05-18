# Python Package for Sky Pixel Detection - Masters Thesis Subproject

#### Author : Rishabh Saxena

## Baseline Mean Shift App

This is an API endpoint that performs baseline mean shift segmentation on an uploaded image and returns the resulting image with the mask applied. It utilizes the FastAPI framework and OpenCV library to process the images.

### Installation

To install and use the `baseline_mean_shift` function, you can follow these installation steps:

1. Clone the repository from GitHub:

   ```bash
   git clone https://github.com/cjblackout/baseline_sky_segmentation.git
   ```

2. Navigate to the project directory:

   ```bash
   cd baseline_sky_segmentation
   ```

3. Create a virtual environment (optional but recommended):

   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

4. Install the required packages using `pip` and the provided `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

   This command will install all the necessary dependencies specified in the `requirements.txt` file.

5. Once the dependencies are installed, you can import and use the `baseline_mean_shift` function in your Python code:

   ```python
   from baseline_mean_shift import baseline_mean_shift

   # Use the function with your image data
   img = cv2.imread('image.jpg')
   mask, segmented_image, labels_image = baseline_mean_shift(img)

   # Perform further operations with the results
   # ...
   ```

   Make sure to replace `'image.jpg'` with the path to your actual image file.

### Usage

To use this API endpoint, send a POST request to the `/baseline_mean_shift_app` endpoint with the following parameters:

- `image` (required): The image file to be processed. Upload it as a file in the request body.
- `processing_shape` (optional): The shape (width, height) to which the image should be resized for processing. Default value is `PROCESSING_SHAPE`.
- `spatial_radius` (optional): The spatial radius parameter for mean shift. Default value is `SPATIAL_RADIUS`.
- `range_radius` (optional): The range radius parameter for mean shift. Default value is `RANGE_RADIUS`.
- `min_density` (optional): The minimum density parameter for mean shift. Default value is `MIN_DENSITY`.

The API will return a streaming response containing the processed image with the mask applied in JPEG format.

#### Example

```python
import requests

url = "http://example.com/baseline_mean_shift_app"
image_file_path = "path/to/image.jpg"

files = {'image': open(image_file_path, "rb")}
response = requests.post(url, files=files)

if response.status_code == 200:
    with open('path/to/segmented_image.jpg', "wb") as f:
        f.write(response.content)
    print("Segmented image saved successfully!")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### Error Handling

If there is an error during image processing, the API will return a JSON response with an `error` field containing the error message.

---

## Baseline Mean Shift Function

This is the `baseline_mean_shift` function that applies baseline mean shift segmentation to an input image and generates a mask for the dominant region (e.g., sky). The function takes the following parameters:

- `img`: The input image to be segmented as a numpy.ndarray.
- `processing_shape`: The shape to resize the input image for processing. It is optional and defaults to `PROCESSING_SHAPE`.
- `spatial_radius`: The

 spatial radius parameter for mean shift segmentation. It controls the size of the spatial neighborhood. It is optional and defaults to `SPATIAL_RADIUS`.
- `range_radius`: The range radius parameter for mean shift segmentation. It controls the size of the color range neighborhood. It is optional and defaults to `RANGE_RADIUS`.
- `min_density`: The minimum density parameter for mean shift segmentation. It controls the minimum number of pixels in a region. It is optional and defaults to `MIN_DENSITY`.

The function returns the following:

- `mask`: The binary mask representing the dominant region (e.g., sky) in the original image shape.
- `segmented_image`: The segmented image with labeled regions.
- `labels_image`: The image with labels corresponding to each region.

The function may raise a `ValueError` if the input image is not a valid numpy.ndarray, and it may raise an `Exception` if an error occurs during mean shift segmentation.

Here is an example usage of the function:

```python
try:
    img = cv2.imread('image.jpg')
    mask, segmented_image, labels_image = baseline_mean_shift(img)
except ValueError as ve:
    print(f"Invalid input image: {ve}")
except Exception as e:
    print(f"Error occurred during mean shift segmentation: {e}")
```