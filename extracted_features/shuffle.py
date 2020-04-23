import os
import random
import sys
 
music_files=[]
 
if len(sys.argv) != 2:
  print("Usage:", sys.argv[0], "/path/directory")
else:
  dir_name=sys.argv[1]
  if os.path.isdir(dir_name):
    for file_name in os.listdir(dir_name):
      music_files.append(file_name)
  else:
    print("Directory", dir_name, "does not exist")
    sys.exit(1)
# shuffle list
random.shuffle(music_files)