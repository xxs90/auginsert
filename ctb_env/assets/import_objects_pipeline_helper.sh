# STEP 1: Model objects (i.e. using Blender) and export their .obj files
#    - Put all .obj files into ctb_env/assets/interactables/meshes/

# STEP 2: Preprocess obj files
#    - Execute this script in ctb_env/assets
python utils/preprocess_obj.py

# STEP 3: Use CoACD to perform convex hull decomposition
#    - Execute this script in ctb_env/assets
#    - Note that the convex_decomp.sh script was used to perform this command on all of the 
#      existing peg/hole objects in this repo
#    - You may need to adjust some parameters depending on the characteristics of your model
#    - This script will dump all of the convex hulls into ctb_env/assets/interactables/meshes/collision_meshes
python utils/convex_hull_decomp.py -i interactables/meshes/${FILENAME}_preprocessed_convex.obj -o interactables/meshes/${FILENAME}_decomp.obj -t 0.04 -pr 100

# STEP 4: Create XML files for the object to import into the simulation
#    - Follow the examples in ctb_env/assets/interactables/*.xml
#    - If your files follow the same general naming scheme of the existing files in the repo
#      (i.e. [PEG/HOLE SHAPE]_[BODY SHAPE]_[CAP/BOTTLE].obj), then you can use the command below 
#      to automatically generate the XML files
#    - Execute this script in ctb_env/assets
python utils/obj_xml_writer.py
