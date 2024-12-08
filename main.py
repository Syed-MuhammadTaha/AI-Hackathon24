# # import os

# # def rename_files_in_folder(folder_path):
# #     # Ensure the folder exists
# #     if not os.path.exists(folder_path):
# #         print(f"Error: Folder '{folder_path}' does not exist.")
# #         return

# #     # Iterate through all files in the folder
# #     for filename in os.listdir(folder_path):
# #         # Check if the file starts with 'frame_'
# #         if filename.startswith('frame_'):
# #             # Create the new filename by removing 'frame_'
# #             new_filename = filename.replace('frame_', '', 1)
            
# #             # Get full paths for renaming
# #             old_file_path = os.path.join(folder_path, filename)
# #             new_file_path = os.path.join(folder_path, new_filename)
            
# #             # Rename the file
# #             os.rename(old_file_path, new_file_path)
# #             print(f"Renamed: {filename} -> {new_filename}")

# #     print("All applicable files have been renamed.")

# # # Example usage
# # folder_path = "/home/user/AI-Hackathon24/data/gtea_png/png/S1_Cheese_C1"  # Replace with the path to your folder
# # rename_files_in_folder(folder_path)


# import xml.etree.ElementTree as ET

# def remove_frame_prefix_from_xml(xml_file, output_file):
#     # Parse the XML file
#     tree = ET.parse(xml_file)
#     root = tree.getroot()
    
#     # Iterate through all elements and attributes in the XML
#     for elem in root.iter():
#         # Check if the element has attributes
#         for attr, value in elem.attrib.items():
#             # If the value contains 'frame_', remove it
#             if isinstance(value, str) and 'frame_' in value:
#                 new_value = value.replace('frame_', '', 1)
#                 elem.set(attr, new_value)

#     # Write the updated XML to the output file
#     tree.write(output_file, encoding='utf-8', xml_declaration=True)
#     print(f"Updated XML file saved to: {output_file}")

# # Example usage
# input_xml_file = "/home/user/AI-Hackathon24/data/xml_labels/S1_Cheese_C1.xml"  # Replace with the path to your XML file
# output_xml_file = "/home/user/AI-Hackathon24/data/xml_labels/S1_Cheese_C1_new.xml"  # Replace with the desired output path
# remove_frame_prefix_from_xml(input_xml_file, output_xml_file)



import os

def rename_images(folder_path):
    # List all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image (you can customize the extensions)
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            # Remove leading zeros from the filename
            new_name = str(int(filename.split('.')[0])) + '.' + filename.split('.')[-1]
            # Get full paths
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_name)
            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed: {old_file} -> {new_file}")

if __name__ == "__main__":
    folder_path = "/home/user/AI-Hackathon24/data/xml_labels/FRAMES/S4_Hotdog_C1"
    rename_images(folder_path)
