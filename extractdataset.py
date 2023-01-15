# importing the "tarfile" module
import tarfile

# open file
file = tarfile.open('./data/atrw_detection_train.tar.gz')
  
# extracting file
file.extractall('./data')
  
file.close()