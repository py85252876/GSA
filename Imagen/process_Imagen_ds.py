import pickle
import random
import time
import torch

def save_dict_to_pkl(dictionary, filename):
    torch.save(dictionary,filename)

def load_dict_from_pkl(filename):
    return torch.load(filename)

def split_and_save(embeddings_and_masks):
    current_time = int(time.time())
    random.seed(current_time)
    print("save target model",flush = True)
    remaining = embeddings_and_masks.copy()
    first_member_keys = random.sample(remaining, 50000)
    first_member_list = [item for item in first_member_keys]
    save_dict_to_pkl(first_member_list, f"./embedding/exp2/target/member.pkl")
    remaining = [item for item in remaining if item not in first_member_keys]
    first_non_member_keys = random.sample(remaining, 5000)
    first_non_member_dict = [item for item in first_non_member_keys]
    remaining = [item for item in remaining if item not in first_non_member_keys]
    save_dict_to_pkl(first_non_member_dict, f"./embedding/exp2/target/non_member.pkl")
    new_remaining = remaining.copy()
    for i in range(6):
        temp = new_remaining.copy()
        print(f"save shadow model{i}", flush=True)
        # Save the non_members dictionary containing 5000 random elements
        non_members_keys = random.sample(new_remaining, 5000)
        non_members_dict = [item for item in non_members_keys]
        save_dict_to_pkl(non_members_dict, f"./embedding/exp2/shadow0{i+1}/non_member.pkl")

        # Save the members dictionary containing the remaining 50000 elements
        new_remaining = [item for item in new_remaining if item not in non_members_keys]
        member_keys = random.sample(new_remaining, 50000)
        members_dict = [item for item in member_keys]
        save_dict_to_pkl(members_dict, f"./embedding/exp2/shadow0{i+1}/member.pkl")
        new_remaining = temp.copy()

def main():
    # Load the embeddings_and_masks dictionary from the .pkl file
    embeddings_and_masks = load_dict_from_pkl("./embedding/text_base_train_1.pkl")

    # Split the dictionary into 1 non_member dictionary with 5000 elements and 1 member dictionary with 50000 elements
    split_and_save(embeddings_and_masks)

if __name__ == '__main__':
    main()