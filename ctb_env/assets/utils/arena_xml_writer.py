from xml.dom import minidom
from glob import glob
import os

def create_and_export_xml(texture):
    root = minidom.Document()

    # Top level
    base = root.createElement('mujoco')
    base.setAttribute('model', 'ctb_arena')
    root.appendChild(base)

    # Assets
    assets = root.createElement('asset')
    base.appendChild(assets)

    # Skybox texture
    sb_t = root.createElement('texture')
    sb_t.setAttribute('builtin', 'gradient')
    sb_t.setAttribute('height', '256')
    sb_t.setAttribute('rgb1', '.9 .9 1.')
    sb_t.setAttribute('rgb2', '.2 .3 .4')
    sb_t.setAttribute('type', 'skybox')
    sb_t.setAttribute('width', '256')
    assets.appendChild(sb_t)

    # Floor texture
    f_t = root.createElement('texture')
    f_t.setAttribute('file', f'textures/{texture}.png')
    f_t.setAttribute('type', '2d')
    f_t.setAttribute('name', 'texplane')
    f_t.setAttribute('width', '300')
    f_t.setAttribute('height', '300')
    assets.appendChild(f_t)

    # Floorplane material
    f_m = root.createElement('material')
    f_m.setAttribute('name', 'floorplane')
    f_m.setAttribute('reflectance', '0.0')
    f_m.setAttribute('shininess', '0.0')
    f_m.setAttribute('specular', '0.0')
    f_m.setAttribute('texrepeat', '2 2')
    f_m.setAttribute('texture', 'texplane')
    f_m.setAttribute('texuniform', 'true')
    assets.appendChild(f_m)

    # Wall texture
    w_t = root.createElement('texture')
    w_t.setAttribute('file', f'textures/light-gray-plaster.png')
    w_t.setAttribute('type', '2d')
    w_t.setAttribute('name', 'tex-light-gray-plaster')
    assets.appendChild(w_t)

    # Wall material
    w_m = root.createElement('material')
    w_m.setAttribute('name', 'walls_mat')
    w_m.setAttribute('reflectance', '0.0')
    w_m.setAttribute('shininess', '0.1')
    w_m.setAttribute('specular', '0.1')
    w_m.setAttribute('texrepeat', '3 3')
    w_m.setAttribute('texture', 'tex-light-gray-plaster')
    w_m.setAttribute('texuniform', 'true')
    assets.appendChild(w_m)

    # Worldbody
    worldbody = root.createElement('worldbody')
    base.appendChild(worldbody)

    # Floor geom
    f_g = root.createElement('geom')
    f_g.setAttribute('condim', '3')
    f_g.setAttribute('group', '1')
    f_g.setAttribute('material', 'floorplane')
    f_g.setAttribute('name', 'floor')
    f_g.setAttribute('pos', '0 0 0')
    f_g.setAttribute('size', '3 3 .125')
    f_g.setAttribute('type', 'plane')
    worldbody.appendChild(f_g)

    # Wall geoms
    poss = [
        '-1.25 2.25 1.5',
        '-1.25 -2.25 1.5',
        '1.25 3 1.5',
        '1.25 -3 1.5',
        '-2 0 1.5',
        '3 0 1.5',
    ]

    quats = [
        "0.6532815 0.6532815 0.2705981 0.2705981",
        "0.6532815 0.6532815 -0.2705981 -0.2705981",
        "0.7071 0.7071 0 0",
        "0.7071 -0.7071 0 0",
        "0.5 0.5 0.5 0.5",
        "0.5 0.5 -0.5 -0.5",
    ]

    sizes = [
        "1.06 1.5 0.01",
        "1.06 1.5 0.01",
        "1.75 1.5 0.01",
        "1.75 1.5 0.01",
        "1.5 1.5 0.01",
        "3 1.5 0.01"
    ]

    names = [
        "wall_leftcorner_visual",
        "wall_rightcorner_visual",
        "wall_left_visual",
        "wall_right_visual",
        "wall_rear_visual",
        "wall_front_visual"
    ]

    for pos, quat, size, name in zip(poss, quats, sizes, names):
        g = root.createElement('geom')
        g.setAttribute('pos', pos)
        g.setAttribute('quat', quat)
        g.setAttribute('size', size)
        g.setAttribute('type', 'box')
        g.setAttribute('conaffinity', '0')
        g.setAttribute('contype', '0')
        g.setAttribute('group', '1')
        g.setAttribute('name', name)
        g.setAttribute('material', 'walls_mat')
        worldbody.appendChild(g)
    
    l = root.createElement('light')
    l.setAttribute('pos', '1.0 1.0 1.5')
    l.setAttribute('dir', '-0.2 -0.2 -1')
    l.setAttribute('specular', '0.3 0.3 0.3')
    l.setAttribute('directional', 'true')
    l.setAttribute('castshadow', 'false')
    worldbody.appendChild(l)
    
    xml_str = root.toprettyxml(indent='\t')
    xml_str = xml_str.split('<?xml version="1.0" ?>\n')[-1]
    save_path = os.path.join('arena', f'ctb_arena_{texture}.xml')

    with open(save_path, 'w') as f:
        f.write(xml_str)
    
    print('Saved', save_path)

for p in glob(os.path.join('arena', 'textures', '*.png')):
    texture = p.split('/')[-1].split('.')[0]
    create_and_export_xml(texture)