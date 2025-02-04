from xml.dom import minidom
from glob import glob
import os

OBJ_TO_ROLE = {
    'cap': 'peg',
    'bottle': 'hole'
}

# NOTE: Run from robosuite_sim_env/assets

def create_and_export_xml(top_shape, body_shape, type):
    root = minidom.Document()

    # Top level
    base = root.createElement('mujoco')
    base.setAttribute('model', 'peg-hole')
    root.appendChild(base)

    # Defaults
    defaults = root.createElement('default')
    base.appendChild(defaults)

    default_vis = root.createElement('default')
    default_vis.setAttribute('class', 'visual')
    defaults.appendChild(default_vis)

    geom_vis = root.createElement('geom')
    geom_vis.setAttribute('type', 'mesh')
    geom_vis.setAttribute('contype', '0')
    geom_vis.setAttribute('conaffinity', '0')
    default_vis.appendChild(geom_vis)

    default_col = root.createElement('default')
    default_col.setAttribute('class', 'collision')
    defaults.appendChild(default_col)

    geom_col = root.createElement('geom')
    geom_col.setAttribute('type', 'mesh')
    geom_col.setAttribute('rgba', '0.5 0.5 0.5 0.0')
    geom_col.setAttribute('density', '100')
    default_col.appendChild(geom_col)

    # Assets (meshes)
    assets = root.createElement('asset')
    base.appendChild(assets)

    # Visual mesh
    mesh = root.createElement('mesh')
    mesh.setAttribute('name', f'{top_shape}_{OBJ_TO_ROLE[type]}_visual')
    mesh.setAttribute('file', f'new_meshes/{top_shape}_{body_shape}_{type}.obj')
    mesh.setAttribute('scale', '1.00 1.00 1.00')
    assets.appendChild(mesh)

    # Collision meshes
    for n in range(len(glob(os.path.join('interactables', 'new_meshes', 'collision_meshes', f'{top_shape}_{body_shape}_{type}*.obj')))):
        mesh = root.createElement('mesh')
        mesh.setAttribute('name', f'{top_shape}_{OBJ_TO_ROLE[type]}_{n}')
        mesh.setAttribute('file', f'new_meshes/collision_meshes/{top_shape}_{body_shape}_{type}_decomp_{n}.obj')
        mesh.setAttribute('scale', '1.00 1.00 1.00')
        assets.appendChild(mesh)

    worldbody = root.createElement('worldbody')
    base.appendChild(worldbody)

    obj_body = root.createElement('body')
    obj_body.setAttribute('name', f'{top_shape}_{OBJ_TO_ROLE[type]}')
    worldbody.appendChild(obj_body)

    mesh_body = root.createElement('body')
    mesh_body.setAttribute('name', 'object')
    obj_body.appendChild(mesh_body)

    for n in range(len(glob(os.path.join('interactables', 'new_meshes', 'collision_meshes', f'{top_shape}_{body_shape}_{type}*.obj')))):
        geom = root.createElement('geom')
        geom.setAttribute('name', f'{top_shape}_{OBJ_TO_ROLE[type]}_{n}')
        geom.setAttribute('class', 'collision')
        geom.setAttribute('friction', '0.01 0.3 0.0001')
        geom.setAttribute('mesh', f'{top_shape}_{OBJ_TO_ROLE[type]}_{n}')
        geom.setAttribute('group', '0')
        mesh_body.appendChild(geom)

    geom = root.createElement('geom')
    geom.setAttribute('name', f'{top_shape}_{OBJ_TO_ROLE[type]}_visual')
    geom.setAttribute('class', 'visual')
    geom.setAttribute('mesh', f'{top_shape}_{OBJ_TO_ROLE[type]}_visual')
    geom.setAttribute('group', '1')
    mesh_body.appendChild(geom)

    for site_name in ['bottom_site', 'top_site', 'horizontal_radius_site']:
        site = root.createElement('site')
        site.setAttribute('rgba', '0 0 0 0')
        site.setAttribute('size', '0.005')
        site.setAttribute('pos', '0 0 0')
        site.setAttribute('name', site_name)
        obj_body.appendChild(site)
    
    xml_str = root.toprettyxml(indent='\t')
    xml_str = xml_str.split('<?xml version="1.0" ?>\n')[-1]
    save_path = os.path.join('interactables', f'{top_shape.lower()}_{body_shape}_{OBJ_TO_ROLE[type]}.xml')

    with open(save_path, 'w') as f:
        f.write(xml_str)
    
    print('Saved', save_path)

for p in glob(os.path.join('interactables', 'new_meshes', '*_preprocessed_convex.obj')):
    ps = p.split('/')[-1].split('_')
    top_shape, body_shape, type = ps[0], ps[1], ps[2]
    create_and_export_xml(top_shape, body_shape, type)