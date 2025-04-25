import os
import pandas as pd
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import xml.etree.ElementTree as ET

def read_beeguard_xml(xml_path, image_folder):
    """
    Parse a BeeGuard XML file containing detection metadata and return a dataframe
    with bounding box information for each image.
    
    Args:
        xml_path (str): Path to the XML file containing detection metadata.
        image_folder (str): Path to the folder containing the images.
        
    Returns:
        pd.DataFrame: DataFrame containing detection information with columns:
            - image_file: Name of the image file
            - width: Image width
            - height: Image height
            - date: Image capture date
            - hour: Image capture time
            - bbox: List of bounding boxes (each with label, score, xtl, ytl, xbr, ybr)
    """
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Initialize lists to store data
    data = []
    
    # Process each image entry in the XML
    for image in root.findall('.//image'):
        # Extract image attributes
        image_name = image.get('name')
        width = int(image.get('width', 0))
        height = int(image.get('height', 0))
        date = image.get('date', '')
        hour = image.get('hour', '')
        
        # Find all bounding boxes for this image
        bbox_list = []
        for box in image.findall('.//box'):
            bbox = {
                'label': box.get('label', ''),
                'score': float(box.get('score', 0)),
                'xtl': float(box.get('xtl', 0)),
                'ytl': float(box.get('ytl', 0)),
                'xbr': float(box.get('xbr', 0)),
                'ybr': float(box.get('ybr', 0))
            }
            bbox_list.append(bbox)
        
        # Get image file path - check in both INSECT and NOTHING subfolders
        insect_path = os.path.join(image_folder, "INSECT", image_name + ".jpg")
        nothing_path = os.path.join(image_folder, "NOTHING", image_name + ".jpg")
        
        # Use the first path that exists, or default to a path that will be checked later
        if os.path.exists(insect_path):
            image_file_path = insect_path
        elif os.path.exists(nothing_path):
            image_file_path = nothing_path
        else:
            # Default path to be checked/corrected later in the code
            image_file_path = os.path.join(image_folder, image_name + ".jpg")
        
        # Extract subfolder (INSECT/NOTHING) from the path
        subfolder = None
        if os.path.exists(insect_path):
            image_file_path = insect_path
            subfolder = "INSECT"
        elif os.path.exists(nothing_path):
            image_file_path = nothing_path
            subfolder = "NOTHING"
        else:
            # Default path to be checked/corrected later in the code
            image_file_path = os.path.join(image_folder, image_name + ".jpg")
            subfolder = "UNKNOWN"
        
        # Add data for this image
        data.append({
            'image_file': image_name,
            'image_path': image_file_path,
            'subfolder': subfolder,
            'width': width,
            'height': height,
            'date': date,
            'hour': hour,
            'bbox': bbox_list
        })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Check if the specified image files exist
        df['file_exists'] = df['image_path'].apply(os.path.exists)
        
        # If some files don't exist, try to find them with different extensions
        if not all(df['file_exists']):
            # Get all images in the directory
            all_images = glob.glob(os.path.join(image_folder, '*.*'))
            all_images += glob.glob(os.path.join(image_folder, 'INSECT', '*.*'))
            all_images += glob.glob(os.path.join(image_folder, 'NOTHING', '*.*'))
            image_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in all_images}
            
            # Update image paths
            for i, row in df[~df['file_exists']].iterrows():
                if row['image_file'] in image_dict:
                    df.at[i, 'image_path'] = image_dict[row['image_file']]
                    # Update subfolder based on the new path
                    if 'INSECT' in image_dict[row['image_file']]:
                        df.at[i, 'subfolder'] = 'INSECT'
                    elif 'NOTHING' in image_dict[row['image_file']]:
                        df.at[i, 'subfolder'] = 'NOTHING'
                    df.at[i, 'file_exists'] = True
        # Drop the helper column
        df = df.drop(columns=['file_exists'])
    
    return df

# Example usage:
xml_path = "examples/Entomoscope_test/inf_entomo2705_sem13/25-03-25/012705_25-03-25T144500.000_P_C0_auto.xml"
image_folder = "/home/edgarremy/Documents/CODE/quick-detector/examples/Entomoscope_test/photo_entomo2705_sem13/25-03-25/"
df = read_beeguard_xml(xml_path, image_folder)
print(df.head())



def display_image_with_bboxes(df, index=0):
    """
    Display an image with its bounding boxes drawn on it.
    
    Args:
        df (pd.DataFrame): DataFrame containing detection information.
        index (int): Index of the image to display. Default is 0 (first image).
    """
    if len(df) <= index:
        print(f"Index {index} is out of bounds for dataframe with {len(df)} rows.")
        return
    
    # Get image and bbox data
    row = df.iloc[index]
    img_path = row['image_path']
    bboxes = row['bbox']
    
    # Open the image
    try:
        img = Image.open(img_path)
        img_np = np.array(img)
        
        # Create figure and axis
        fig, ax = plt.subplots(1, figsize=(12, 8))
        
        # Display the image
        ax.imshow(img_np)
        
        # Draw bounding boxes
        for bbox in bboxes:
            # Get coordinates
            xtl = bbox['xtl']
            ytl = bbox['ytl']
            width = bbox['xbr'] - bbox['xtl']
            height = bbox['ybr'] - bbox['ytl']
            
            # Create rectangle
            rect = patches.Rectangle(
                (xtl, ytl), width, height,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            
            # Add rectangle to plot
            ax.add_patch(rect)
            
            # Add label and score
            ax.text(
                xtl, ytl - 5, 
                f"{bbox['label']} ({bbox['score']:.2f})", 
                color='red', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7)
            )
        
        # Set title with image info
        ax.set_title(f"Image: {row['image_file']} - {row['date']} {row['hour']}")
        
        # Turn off axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error displaying image: {e}")

# Display the first image with bounding boxes
for i in range(len(df)):
    display_image_with_bboxes(df, index=i)
# display_image_with_bboxes(df, index=0)
# display_image_with_bboxes(df, index=1)
# display_image_with_bboxes(df, index=2)

