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
    for demo in glob(os.path.join(args.dataset, "*.h5")):
        f_demo = h5py.File(demo, "r")
        if len(f_demo['actions'][:]) > 0:
            demo_grp = grp.create_group(f"demo_{n_demo}")

            # TODO: [IMPORTANT] Scale down output actions by 1000 during rollouts
            demo_grp.create_dataset("actions", data=f_demo['actions'][:][:,[0,2]]*1000.0) 
            demo_grp.create_dataset("obs/overhead_image", data=f_demo['obs/overhead_image'][:])

            # Normalize depth (will also need to do this during rollouts as well)
            demo_grp.create_dataset("obs/overhead_depth", data=np.expand_dims(np.clip(f_demo['obs/overhead_depth'][:] / 6000.0, 0.0, 1.0), axis=-1))

            # Concatenate force-torque observations
            ft_data = np.concatenate([
                f_demo['obs/robot0_forcetorque'][:],
                f_demo['obs/robot1_forcetorque'][:]
            ], axis=-1)
            demo_grp.create_dataset("obs/robot0_robot1_forcetorque-state", data=ft_data)

            # Concatenate proprioception observations
            prop_data = np.concatenate([
                f_demo['obs/robot0_eef_pos'][:],
                f_demo['obs/robot0_eef_quat'][:],
                f_demo['obs/robot0_eef_pos'][:],
                f_demo['obs/robot0_eef_quat'][:]
            ], axis=-1)
            demo_grp.create_dataset("obs/robot0_robot1_proprioception-state", data=prop_data)

            n_sample = f_out[f"data/demo_{n_demo}/actions"].shape[0]
            f_out[f"data/demo_{n_demo}"].attrs["num_samples"] = n_sample
            total_samples += n_sample
            
            n_demo += 1
    
    f_out["data"].attrs["total"] = total_samples
    f_out.close()

    # create 90-10 train-validation split in the dataset
    split_train_val_from_hdf5(hdf5_path=args.hdf5_name, val_ratio=5/n_demo)

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