from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
import os
from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy as np

class ObjectIdentifier:
    def __init__(self, model_name="microsoft/resnet-50"):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ResNetForImageClassification.from_pretrained(model_name)
    
    def identify_objects_from_bboxes(self, output_dir, master_id, ipath):
        # Load metadata
        metadata_filename = os.path.join(output_dir, f"{master_id}_metadata.json")
        with open(metadata_filename, 'r') as f:
            metadata = json.load(f)

        # Load the original image
        original_image = Image.open(ipath)

        # Iterate over each entry in the metadata
        for entry in metadata:
            object_id = entry["object_id"]
            bbox = entry["bbox"]
            
            # Crop the original image using the bounding box
            x, y, width, height = bbox
            cropped_image = original_image.crop((x, y, x + width, y + height))
            
            # Preprocess the image
            inputs = self.processor(cropped_image, return_tensors="pt")

            # Predict the class
            with torch.no_grad():
                logits = self.model(**inputs).logits
            predicted_label = logits.argmax(-1).item()
            description = self.model.config.id2label[predicted_label]
            
            # Update metadata with description
            entry['description'] = description

            # Optionally display the cropped image and prediction
            plt.figure(figsize=(6, 6))
            plt.imshow(np.array(cropped_image))
            plt.title(f"Object ID: {object_id}\nPrediction: {description}")
            plt.axis('off')
            plt.show()

        # Save updated metadata
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=4)

        return metadata


# Example usage
output_dir = "/path/to/output/directory"
master_id = "image_01"
ipath = "/path/to/original/image.jpg"

identifier = ObjectIdentifier(model_name="microsoft/resnet-50")
updated_metadata = identifier.identify_objects_from_bboxes(output_dir, master_id, ipath)

print(updated_metadata)
