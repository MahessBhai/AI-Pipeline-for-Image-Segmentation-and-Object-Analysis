import streamlit as st
from PIL import Image
import os
import json
from models.ocr_seg_iden_model import ObjectDetectionPipeline

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
                    master_id = pipeline.process(img_path, output_dir)
                    st.success(f"Processing completed! Results are saved in {output_dir}")

                    # Show the segmented image with master_id
                    segmented_image_path = os.path.join(output_dir, "output", f"{master_id}_segmented.png")
                    if os.path.exists(segmented_image_path):
                        st.image(segmented_image_path, caption="Segmented Image")
                    else:
                        st.warning("Segmented image not found.")

                    # Option to view segmented objects and metadata
                    if st.checkbox("Show Segmented Objects"):
                        segmented_objects_dir = os.path.join(output_dir, "segmented_objects", master_id)
                        if os.path.exists(segmented_objects_dir):
                            segmented_files = os.listdir(segmented_objects_dir)
                            for obj_file in segmented_files:
                                st.image(os.path.join(segmented_objects_dir, obj_file), caption=obj_file)
                        else:
                            st.warning("Segmented objects not found.")

                    if st.checkbox("Show Metadata"):
                        metadata_file = os.path.join(output_dir, "output", f"{master_id}_metadata.json")
                        if os.path.exists(metadata_file):
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            st.json(metadata)
                        else:
                            st.warning("Metadata not found.")

            else:
                if input_file:
                    for image_file in input_file:
                        img_path = os.path.join(output_dir, image_file.name)
                        # Save each uploaded file to a temp location
                        with open(img_path, "wb") as f:
                            f.write(image_file.getbuffer())
                        
                        # Process each saved image
                        st.write(f"Processing {img_path}...")
                        master_id = pipeline.process(img_path, output_dir)
                    
                    st.success(f"Processing completed for all images! Results are saved in {output_dir}")

if __name__ == "__main__":
    main()
