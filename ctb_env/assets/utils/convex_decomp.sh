for BODY in 'octagonal' 'cube-thin' 'cylinder-thin' 'octagonal-thin'
do
    for SHAPE in 'Arrow' 'Circle' 'Cross' 'Diamond' 'Hexagon' 'Key' 'Line' 'Pentagon' 'U'
    do
        python utils/convex_hull_decomp.py -i interactables/meshes/ctb_objs/${SHAPE}_${BODY}_bottle_preprocessed_convex.obj -o interactables/meshes/${SHAPE}_${BODY}_bottle_decomp.obj -t 0.04 -pr 100
        python utils/convex_hull_decomp.py -i interactables/meshes/ctb_objs/${SHAPE}_${BODY}_cap_preprocessed_convex.obj -o interactables/meshes/${SHAPE}_${BODY}_cap_decomp.obj -t 0.04 -pr 100
    done
done
