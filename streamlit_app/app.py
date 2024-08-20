import streamlit as st
from PIL import Image
import os
import json
from models.ocr_seg_iden_model import ObjectDetectionPipeline

# pipeline = ObjectDetectionPipeline()


def main():
    # Title and description
    st.title("Image Segmentation and Object Analysis")
    st.write("This pipeline processes images to detect and segment objects, performs OCR, and stores the results.")

    # Sidebar for input options
    st.sidebar.title("Options")
    
    # File uploader for a single image or multiple images
    input_type = st.sidebar.radio("Input Type", ('Single Image', 'Folder of Images'))
    
    if input_type == 'Single Image':
        input_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    else:
        input_file = st.sidebar.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    # Output directory input
    output_dir = st.sidebar.text_input("Output Directory", value=os.getcwd())

    # Initialize the pipeline
    pipeline = ObjectDetectionPipeline()

    # Process button
    if st.sidebar.button("Process"):
        if not input_file:
            st.error("Please upload an image or select a folder.")
        else:
            if input_type == 'Single Image':
                if input_file is not None:
                    # Save the uploaded file to a temp location
                    img_path = os.path.join(output_dir, input_file.name)
                    with open(img_path, "wb") as f:
                        f.write(input_file.getbuffer())
                    
                    # Process the saved image
                    st.write(f"Processing {img_path}...")
                    pipeline.process(img_path, output_dir)
                    st.success(f"Processing completed! Results are saved in {output_dir}")

            else:
                if input_file:
                    for image_file in input_file:
                        img_path = os.path.join(output_dir, image_file.name)
                        # Save each uploaded file to a temp location
                        with open(img_path, "wb") as f:
                            f.write(image_file.getbuffer())
                        
                        # Process each saved image
                        st.write(f"Processing {img_path}...")
                        pipeline.process(img_path, output_dir)
                    
                    st.success(f"Processing completed for all images! Results are saved in {output_dir}")

if __name__ == "__main__":
    main()
