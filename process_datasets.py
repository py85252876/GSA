import random
import os
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        required=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True
    )
    parser.add_argument(
        "--datanum_target_model",
        type=int,
        default=None,
        required=True
    )
    parser.add_argument(
        "--datanum_per_shadow_model",
        type=int,
        default=None,
        required=True
    )
    parser.add_argument(
        "--number_of_shadow_model",
        type=int,
        default=None,
        required=True
    )
    args = parser.parse_args()
    return args

def get_file_path(filepath):
    files = os.listdir(filepath)   
    file_path_list = []
    print("Starting select file")
    for fi in files:    
        fi_d = os.path.join(filepath,fi)    
        if os.path.isdir(fi_d):
            full_path = os.path.join(filepath, fi_d)
            # file_path_list.append(full_path)
            file_path_list.extend(get_file_path(fi_d))    
        else:      
            full_path = os.path.join(filepath,fi_d)
            file_path_list.append(full_path)
    return file_path_list

def random_select(file_list, num_selected):
    if(num_selected > len(file_list)):
        print(f"ERROR: {num_selected} > {len(file_list)}")
        exit(0)
    selected = random.sample(file_list, num_selected)
    not_selected = []
    for i in file_list:
        if(i not in selected):
            not_selected.append(i)
    return selected, not_selected

def get_shadow_dataset(number_of_shadow_model, shadow_dataset, datanum_per_shadow_model):
    shadow_member = []
    shadow_non_member = []
    print("select shadow dataset")
    for i in range(number_of_shadow_model):
        shadow_data, _ = random_select(shadow_dataset, int(datanum_per_shadow_model))
        shadow_data_member, shadow_data_non_member = random_select(shadow_data, int(len(shadow_data) / 2))
        shadow_member.append(shadow_data_member)
        shadow_non_member.append(shadow_data_non_member)
    return shadow_member, shadow_non_member

def cp_path(target_member,target_non_member, shadow_member, shadow_non_member, output_dir ):
    if(not os.path.exists(output_dir)):
        os.mkdir(output_dir)
    else:
        if(os.listdir(output_dir) != []):
            print(f"ERROR: Please clear {output_dir}.")
            exit(0)
    target_member_path = os.path.join(output_dir, "target_model", "model_member")
    target_non_member_path = os.path.join(output_dir, "target_model", "non_model_member")
    os.makedirs(target_member_path)
    os.makedirs(target_non_member_path)
    print("Copying target model") 
    for file in target_member:
        shutil.copy(file, target_member_path)
        
    for file in target_non_member:
        shutil.copy(file, target_non_member_path)
        
    for i in range(len(shadow_member)):
        print(f"Copying shadow member {i}")
        shadow_member_path = os.path.join(output_dir,"shadow_model",f"{i}", "model_member")
        shadow_non_member_path = os.path.join(output_dir,"shadow_model",f"{i}", "non_model_member")
        os.makedirs(shadow_member_path)
        os.makedirs(shadow_non_member_path)
        for file in shadow_member[i]:
            shutil.copy(file, shadow_member_path)
            
        for file in shadow_non_member[i]:
            shutil.copy(file, shadow_non_member_path)
    print("finish all works")

            

def main():
    args = parse_args()
    file_list = get_file_path(args.dataset_dir)
    target, shadow = random_select(file_list, args.datanum_target_model)
    target_member, target_non_member = random_select(target, int(len(target) / 2))
    shadow_member, shadow_non_member = get_shadow_dataset(args.number_of_shadow_model, shadow, args.datanum_per_shadow_model)
    print("Selected data finished!")
    cp_path(target_member,target_non_member, shadow_member, shadow_non_member, args.output_dir )
    
        
if __name__ == "__main__":
    main()
