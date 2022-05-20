#File for saving curves from yaml file representing MIOU and FID
import yaml
import os
import matplotlib.pyplot as plt

folder_yaml_name = "improved_nightdrive/curves/yaml_files"      #folder in which there is yaml files
folder_curves_name = "improved_nightdrive/curves/curves"        #folder where to save curves
fid_reference = 20                                              #typical "zero-like fid", fid between a distribution and itself (sampled differently) for reference. Having a FID of this value is considered perfect.

def return_data(filename):
    with open(filename, 'r') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        dico = yaml.load(file, Loader=yaml.FullLoader)
        for key, value in dico["Results"].items():
            n_batch = key[-6:]
            fid = value["FID"]
            miou = value["miou"]
        return n_batch, fid, miou

if __name__ == '__main__':
    L_batch = list()
    L_fid = list()
    L_miou = list()
    for filename in os.listdir(folder_yaml_name):
        if filename.endswith(".yaml"):
            n_batch, fid, miou = return_data(os.path.join(folder_yaml_name, filename))
            L_batch.append(n_batch)
            L_fid.append(fid)
            L_miou.append(miou)    
    
    plt.plot(L_batch, L_fid, label="FID")
    plt.plot(L_batch, [fid_reference for _ in L_batch], label = "FID reference")
    plt.ylim(bottom = 0)
    plt.xlabel("n_batch")
    plt.legend()
    plt.savefig(os.path.join(folder_curves_name, "FID.png"))
    
    plt.figure()
    plt.plot(L_batch, L_miou, label="MIOU")
    plt.xlabel("n_batch")
    plt.ylim((0,1))
    plt.legend()
    plt.savefig(os.path.join(folder_curves_name, "MIOU.png"))
    
    