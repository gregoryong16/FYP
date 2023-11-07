import os
import shutil
# importing the "tarfile" module
import tarfile
# ## read and change <filename> field in xml files
import xml.etree.ElementTree as ET

# ##############################################################################
# ## Extracting tar file
# # open file
# file = tarfile.open('atrw_anno_detection_train.tar.gz')
  
# # extracting file
# file.extractall('./')
  
# file.close()
# ##############################################################################
# # to rename <filename> tag in annotations file

# for file in os.listdir('./train_annotations'):
#     # with open("./val_annotations/"+file, encoding='utf8') as f:
#     #     data = f.read()
#     #     # Passing the stored data inside
#     #     # the beautifulsoup parser, storing
#     #     # the returned object
#     #     Bs_data = BeautifulSoup(data, "xml") 
        
#     #     filename, ext = os.path.splitext(file)
#     #     # Finding all instances of tag
#     #     # `unique`
#     #     tag = Bs_data.find_all('filename')
#     #     print(tag)
#     #     tag[0].string.replace_with(filename + ".jpg")   
        
#     #     with open("./val_annotations/"+file, "wb") as file:
#     #         file.write(str(Bs_data))
#     mytree = ET.parse('./train_annotations/'+file)
#     myroot = mytree.getroot()
 
#     # iterating through the price values.
#     for prices in myroot.iter('filename'):
#         # updates the price value
#         filename, ext = os.path.splitext(file)
#         prices.text = str(filename+".jpg")
    
#     mytree.write('./train_annotations/'+file)

###############################################################
## splitting file into train and val 

# count=0
# for file in os.listdir("./train_annotations"):
#     count+=1

# print("Total: ", count)

# ## split the files into train and val annotations (70/30)
# count2 = int(count/100 * 30 )
# print(count2)
# print(count-count2)

# for file in os.listdir("./train_annotations"):
#     if count2 != 0:
#         dest = "./val_annotations"+ "/" + file
#         src = "./train_annotations"+ "/" + file
#         shutil.move(src,dest)
#         count2-=1
        
################################################################
# with open('./val_annpaths_list.txt', 'w') as f:
#     for file in os.listdir("./val_annotations"):
#         print(file)
#         f.write("./val_annotations/"+ file +"\n")

# with open('./train_annpaths_list.txt', 'w') as f:
#     for file in os.listdir("./train_annotations"):
#         print(file)
#         f.write("./train_annotations/"+ file +"\n")

# with open('./dataset_ids/val.txt', 'w') as f:
#     for file in os.listdir("./val_annotations"):
#         path, ext = os.path.splitext(file)
#         f.write(path+"\n")

import os
import shutil

# Source and destination folders
source_folder = 'val_images'
destination_folder = 'test_images'

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Read the list of file names from file_list.txt
with open('test_annpaths_list.txt', 'r') as file:
    file_names = file.read().splitlines()

# Move files from the source folder to the destination folder
for file_name in file_names:
    source_path = os.path.join(source_folder, file_name)
    destination_path = os.path.join(destination_folder, file_name)

    try:
        shutil.move(source_path, destination_path)
        print(f"Moved {file_name} to {destination_folder}")
    except FileNotFoundError:
        print(f"{file_name} not found in {source_folder}")
    except shutil.Error as e:
        print(f"Error moving {file_name}: {e}")
