import cv2
import easyocr
import matplotlib.pyplot as plt
import os

def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)

    # Apply morphological operations if needed (you can comment this out if not needed)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morphed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

    return morphed

def extract_text_from_images(input_dir, output_dir):
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each image in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            # Preprocess the image
            preprocessed_image = preprocess_image(image)

            # Perform OCR on the original and preprocessed images
            results_original = reader.readtext(image)
            results_preprocessed = reader.readtext(preprocessed_image)

            # Combine results
            results_combined = results_original + results_preprocessed

            # Store the results in a text file
            output_text_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_text.txt")
            with open(output_text_path, 'w') as text_file:
                for (bbox, text, prob) in results_combined:
                    text_file.write(f"{text} (Confidence: {prob:.2f})\n")

            # Display the processed image with detected text (optional)
            for (bbox, text, prob) in results_combined:
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))
                cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

            # Save or display the annotated image
            output_image_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_annotated.png")
            cv2.imwrite(output_image_path, image)
            
            # plt.figure(figsize=(10, 10))
            # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.show()

# Specify the input directory containing images and the output directory for results
input_dir = r"C:\Users\adity\Documents\Cheetahs\wasserstoff\project_root\data\segmented_objects\3d8285b1-9c76-4688-9333-6c41e57b14df"
output_dir = 'output'

# Call the function to process the directory of images
extract_text_from_images(input_dir, output_dir)
