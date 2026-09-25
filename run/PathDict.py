import os 

dataroot = "./dataroot"
CodePath = "./CodePath"
ExpFolder = "./ExpFolder"
pretrain_folder = "./pretrain"
cache_root = "./cache"

dataset_dict = {
    "Internal": {
        "train_label"   :  os.path.join(dataroot, "train_center1.csv"),  
        "train_path"    :  os.path.join(dataroot, "center1/"),      
        "val_label"     :  os.path.join(dataroot, "valid_center1.csv"),  
        "val_path"      :  os.path.join(dataroot, "center1/"),      
        "test_label"    :  os.path.join(dataroot, "test_center1.csv"),   
        "test_path"     :  os.path.join(dataroot, "center1/"),      
        "cache_path"    :  os.path.join(cache_root, "center1"),
        "center_file"   :  os.path.join(dataroot, "center.json"),
        "doctor_file"   :  os.path.join(dataroot, "doc_eval.csv"),
        "modal"         :  ["sag PDW_spair","cor PDW_spair","axi T2w_tse","sag T1w_tse","sag PDW_atse"]
    },
    "center4": {      
        "test_label"    :  os.path.join(dataroot, "test_center4.csv"),   
        "test_path"     :  os.path.join(dataroot, "center4/"),      
        "cache_path"    :  os.path.join(cache_root, "center4"),
        "center_file"   :  os.path.join(dataroot, "center.json"),
        "doctor_file"   :  os.path.join(dataroot, "doc_eval.csv"),
        "modal"         :  ["sag PDW_spair","cor PDW_spair","axi T2w_tse","sag T1w_tse","sag PDW_atse"]
    },
    "center3": {
        "test_label"    :  os.path.join(dataroot, "test_center3.csv"),   
        "test_path"     :  os.path.join(dataroot, "center3/"),      
        "cache_path"    :  os.path.join(cache_root, "center3"),
        "center_file"   :  os.path.join(dataroot, "center.json"),
        "doctor_file"   :  os.path.join(dataroot, "doc_eval.csv"),
        "modal"         :  ["sag PDW_spair","cor PDW_spair","axi T2w_tse","sag T1w_tse","sag PDW_atse"] 
    },
}