import h5py
import argparse
import os
import numpy as np
from glob import glob
from robomimic.scripts.split_train_val import split_train_val_from_hdf5

def main(args):
    f_out = h5py.File(args.hdf5_name, "w")
    grp = f_out.create_group("data")
    f_out["data"].attrs["env_args"] = "{}"

    n_demo = 1
    total_samples = 0
    f_demo = h5py.File(args.dataset, "r")
    print(f_demo.keys())
    demo_grp = grp.create_group(f"demo_{n_demo}")

    # TODO: [IMPORTANT] Scale down output actions by 250 during rollouts
    # print(f_demo['obs'].keys())
    # actions = np.clip(f_demo['actions'][:][:,[0,2]]*250.0, -1.0, 1.0) # DELTA ACTIONS
    # actions = np.concatenate((f_demo['obs/robot1_eef_pos'][:][1:,:,[0,2]], np.zeros((1,1,2))), axis=0).squeeze(1) # ABSOLUTE ACTIONS
    # demo_grp.create_dataset("actions", data=actions)
    demo_grp.create_dataset("obs/left_wristview_image", data=np.expand_dims(f_demo['left_wristview_image'][:][...,::-1].transpose(0,2,3,1), 1))
    demo_grp.create_dataset("obs/right_wristview_image", data=np.expand_dims(f_demo['right_wristview_image'][:][...,::-1].transpose(0,2,3,1), 1))

    # Normalize depth (will also need to do this during rollouts as well)
    # demo_grp.create_dataset("obs/overhead_depth", data=np.expand_dims(np.clip(f_demo['obs/robot1_rgb'][:] / 6000.0, 0.0, 1.0), axis=-1))

    # Concatenate force-torque observations
    demo_grp.create_dataset("obs/robot0_robot1_forcetorque-state", data=np.expand_dims(f_demo['robot0_robot1_forcetorque-state'][:], 1))
    for f in f_demo['robot0_robot1_forcetorque-state'][:]:
        print(f_demo['robot0_robot1_forcetorque-state'][:].shape)
        print('FT:', f[-1,:])

    # Concatenate proprioception observations
    demo_grp.create_dataset("obs/robot0_robot1_proprioception-state", data=np.expand_dims(f_demo['robot0_robot1_proprioception-state'][:], 1))

    # n_sample = f_out[f"data/demo_{n_demo}/actions"].shape[0]
    # f_out[f"data/demo_{n_demo}"].attrs["num_samples"] = n_sample
    # total_samples += n_sample
    
    n_demo += 1
    
    f_out["data"].attrs["total"] = total_samples
    f_out.close()

    # create 90-10 train-validation split in the dataset
    # split_train_val_from_hdf5(hdf5_path=args.hdf5_name, val_ratio=5/n_demo)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to input dataset folder",
    )

    parser.add_argument(
        "--hdf5_name",
        type=str,
        default="data.hdf5",
        help="name of output hdf5 file"
    )

    args = parser.parse_args()
    main(args)