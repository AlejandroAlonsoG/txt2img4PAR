import os
import re
import shutil
import numpy as np
from easydict import EasyDict
import time
import sys
import pickle

"""
This script performs data augmentation on hs-BaldHead, lb-ShortSkirt and AgeLss16 attributes of the RAP datasets. Any other attribute works the same way

Usage:
    1. Ensure all required files are in place:
       - Original pickle files: 'original_rap1_dataset_all.pkl', 'original_rap2_dataset_all.pkl',
         and 'original_rap2_dataset_zs.pkl' (located in the designated dataset folder).
       - A subdirectory at {DATASET_PICKLE_PATH}/text_files containing text files that map
         wildcards to their indirect labels. (Note they might differe on each dataset)
       - Images and prompt logs must be in the same folder with matching prefix names.
    2. Configure the parameters below as needed (e.g., dataset_name, attribute, augmentation levels).
    3. Run the script from the command line. The script will prompt for any additional
       file name postfix or image prefix filtering.
    4. The augmented dataset will be saved to a new pickle file in the specified output folder.

"""

#-- Parameters --

attribute = 'bald'
dataset_name = 'RAP2'
COMFYUI_OUTPUT = f'./artifacts/full_bald_augmentation_clean:v0'

PREFIX_IMAGE = f"ComfyUI_{attribute.capitalize()}_123456789"
AUGMENTATION_PERCENTAGE_OPTIONS = [50, 100, 200, 300, 400, 500]

# ----

IMAGE_COUNTS = {
    "RAPzs": {
        "hs-BaldHead": {"train": 116, "test": 6},
        "lb-ShortSkirt": {"train": 643, "test": 144},
        "AgeLess16": {"train": 195, "test": 68},
    },
    "RAP2": {
        "hs-BaldHead": {"train": 391, "test": 92},
        "lb-ShortSkirt": {"train": 1390, "test": 354},
        "AgeLess16": {"train": 603, "test": 150},
    },
    "RAP1": {
        "hs-BaldHead": {"train": 122, "test": 36},
        "lb-ShortSkirt": {"train": 912, "test": 252},
        "AgeLess16": {"train": 334, "test": 81},
    },
}

if dataset_name == "RAP1":
    PICKELFILE = 'original_rap1_dataset_all.pkl'
    labels = "labels"
elif dataset_name == "RAP2":
    PICKELFILE = 'original_rap2_dataset_all.pkl'
    labels = "labels_2"
elif dataset_name == "RAPzs":
    PICKELFILE = 'original_rap2_dataset_zs.pkl'
    labels = "labels_zs"
else:
    raise ValueError(f"Invalid case: {dataset_name}")

if attribute == "bald":
    attribute_name = "hs-BaldHead"
elif attribute == "shortskirt":
    attribute_name = "lb-ShortSkirt"
elif attribute == "young":
    attribute_name = "AgeLess16"
else:
    raise ValueError(f"Invalid case: {attribute}")


DATASET_PICKLE_PATH = './augmentate_dataset'
READ_DATASET_FILE_NAME = f'{PICKELFILE}'

WILDCARDS_WITH_LABELS_DIRECTORY = f'{DATASET_PICKLE_PATH}/text_files'
LABELS_PATH = f'{WILDCARDS_WITH_LABELS_DIRECTORY}/{labels}.txt'
SAVE_DATASET_FILE_NAME = f'{DATASET_PICKLE_PATH}/dataset_all_syn.pkl'

default_image_prefix = ""

def get_user_inputs():
    return {
        "dataset_name": dataset_name,
        "attribute_name": attribute_name,
        "noise_level": "medium",
    }

def get_wildcard_label_mapping():
    mapping = {}

    for filename in os.listdir(WILDCARDS_WITH_LABELS_DIRECTORY):
        if filename.endswith('_labels.txt'):

            if attribute == "shortskirt" and filename == "styles_labels.txt":
                filename = "styles_labels_shortskirt.txt"

            if dataset_name != "RAP1" and filename != "attributes_labels.txt":
                filename = filename[:-4] + "_rap2.txt"

            with open(os.path.join(WILDCARDS_WITH_LABELS_DIRECTORY, filename), 'r') as f:
                for line in f:
                    parts = re.split(r'\[|\]', line.strip())
                    sentence = parts[0].strip()

                    if len(parts) > 1:
                        labels = [label.strip() for label in parts[1].split(',') if '/' not in label]
                        mapping[sentence] = labels

    return mapping

def get_labels_from_prompt(input_string):

    mapping = get_wildcard_label_mapping()
    
    with open(LABELS_PATH, 'r') as f:
        labels_list = [line.strip() for line in f]
    
    label_vector = [-1] * len(labels_list)
    
    for sentence, labels in mapping.items():
        if sentence in input_string:
            for label in labels:
                index = labels_list.index(label)
                label_vector[index] = 3

    return label_vector


def load_dataset(pickle_path):
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
        
def save_dataset(dataset, pickle_path):
    with open(pickle_path, 'wb') as f:
        pickle.dump(dataset, f)
    print("Dataset saved successfully.")

def generate_dataset_report(dataset, labels_from_file):
    """Generate a report comparing the dataset and labels.txt."""
    print("=== Dataset Report ===")

    print(f"Root Directory: {dataset.root}")

    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            labels_from_file = [line.strip() for line in f if line.strip()]
    else:
        print(f"Label file not found at {LABELS_PATH}.")
        return

    dataset_labels = dataset.attr_name if hasattr(dataset, 'attr_name') else []
    
    print(f"Number of labels in dataset: {len(dataset_labels)}")
    print(f"Number of labels in labels.txt: {len(labels_from_file)}")

    missing_in_dataset = [label for label in labels_from_file if label not in dataset_labels]
    additional_in_dataset = [label for label in dataset_labels if label not in labels_from_file]

    if missing_in_dataset:
        print("\nLabels missing in dataset (present in labels.txt):")
        for label in missing_in_dataset:
            print(f"  - {label}")
    else:
        print("\nNo missing labels in the dataset.")

    if additional_in_dataset:
        print("\nAdditional labels in dataset (not in labels.txt):")
        for label in additional_in_dataset:
            print(f"  - {label}")
    else:
        print("\nNo additional labels in the dataset.")

    print("\n=== End of Report ===")
    if len(missing_in_dataset) == 0:
        return labels_from_file
    return 0

def get_max_images(dataset_name, attribute_name, partition="train"):
    if dataset_name not in IMAGE_COUNTS:
        raise ValueError(f"Dataset '{dataset_name}' not found.")
    
    dataset_attributes = IMAGE_COUNTS[dataset_name]
    if attribute_name not in dataset_attributes:
        raise ValueError(f"Attribute '{attribute_name}' not found in dataset '{dataset_name}'.")
    
    if partition not in dataset_attributes[attribute_name]:
        raise ValueError(f"Partition '{partition}' not found for attribute '{attribute_name}' in dataset '{dataset_name}'.")
    
    return dataset_attributes[attribute_name][partition]

def add_image_dataset(image_path, label_vector, dataset, labels_from_file,current_img_num,total_images,unique_name=True):
    original_image_name = os.path.basename(image_path)

    check = False

    dest_path = os.path.join(dataset.root, original_image_name)
    if os.path.exists(dest_path):
        print(f"File already exists in destination folder: {original_image_name}. Adding to dataset without moving.")
        new_image_name = original_image_name
    else:
        if unique_name:
            file_name, file_extension = os.path.splitext(original_image_name)
            unique_id = str(int(time.time()))
            new_image_name = f"{file_name}_{unique_id}{file_extension}"
        else:
            new_image_name = original_image_name

        dest_path = os.path.join(dataset.root, new_image_name)
        check = True

        shutil.copy(image_path, dest_path)
        print(f"Copied image: {image_path} to {dest_path}")

    dataset_labels = dataset.attr_name if hasattr(dataset, 'attr_name') else []
    aligned_label_vector = [-1] * len(dataset_labels)

    for i, label in enumerate(labels_from_file):
        if label in dataset_labels:
            index = dataset_labels.index(label)
            aligned_label_vector[index] = label_vector[i]

    dataset.image_name.append(new_image_name)
    if len(dataset.label[0]) < len(aligned_label_vector):
        extra_cols = len(aligned_label_vector) - len(dataset.label[0])
        dataset.label = np.column_stack((dataset.label, -1 * np.ones((dataset.label.shape[0], extra_cols), dtype=int)))
    dataset.label = np.row_stack((dataset.label, aligned_label_vector))

    new_index = len(dataset.image_name) - 1
    trainval = dataset.partition['trainval']
    for i, percentage in enumerate(AUGMENTATION_PERCENTAGE_OPTIONS):
        required_images = ( percentage * total_images)
        if (current_img_num < required_images):
            trainval[i] = np.append(trainval[i], new_index)

    dataset.partition['trainval'] = trainval

    if 'synth' not in dataset.partition:
        dataset.partition['synth'] = []

    dataset.partition['synth'].append(new_index)

    return check
def update_trainval_partitions(dataset, partition_key='trainval', required_partitions=None):
    if partition_key not in dataset.partition:
        print(f"Partition '{partition_key}' does not exist in the dataset.")
        return

    if dataset_name != "RAPzs":
        first_partition = dataset.partition[partition_key][0]
    else:
        dataset.partition[partition_key] = [dataset.partition[partition_key]] * 5
        first_partition = dataset.partition[partition_key][0]

    print(f"First partition size: {len(first_partition)}")

    for i in range(len(dataset.partition[partition_key])):
        dataset.partition[partition_key][i] = first_partition.copy()
        print(f"Copied first partition into {partition_key}[{i}].")

    current_partitions = len(dataset.partition[partition_key])
    if required_partitions and required_partitions > current_partitions:
        for _ in range(current_partitions, required_partitions):
            dataset.partition[partition_key].append(first_partition.copy())
            print(f"Added new partition to '{partition_key}'.")
        print(f"Expanded {partition_key} to size: {len(dataset.partition[partition_key])}")

    print(f"Updated all partitions under '{partition_key}' with the first partition.")

def filter_images_by_prefix(files, prefix):
    return [file for file in files if file.startswith(prefix) and file.endswith('.png')]


def expand_trainval_partitions(dataset, required_partitions, partition_key='trainval'):
    if partition_key not in dataset.partition:
        print(f"Partition '{partition_key}' does not exist in the dataset.")
        return

    current_partitions = len(dataset.partition[partition_key])
    print(f"Current number of {partition_key} partitions: {current_partitions}")

    if current_partitions >= required_partitions:
        print(f"No need to expand. Current partitions ({current_partitions}) >= required ({required_partitions}).")
        return

    first_partition = dataset.partition[partition_key][0]
    for _ in range(current_partitions, required_partitions):
        dataset.partition[partition_key].append(first_partition.copy())
        print(f"Added new partition to '{partition_key}'.")

    print(f"Updated {partition_key} partitions to size: {len(dataset.partition[partition_key])}")
if __name__ == '__main__':

    current_img_num =0

    for i,per in enumerate(AUGMENTATION_PERCENTAGE_OPTIONS):
        AUGMENTATION_PERCENTAGE_OPTIONS[i] = per/100

    dataset = load_dataset(os.path.join(DATASET_PICKLE_PATH, READ_DATASET_FILE_NAME))

    labels_from_file = generate_dataset_report(dataset, LABELS_PATH)
    if not labels_from_file:
        raise Exception("Dataset labels are not balanced")
    
    required_partitions = len(AUGMENTATION_PERCENTAGE_OPTIONS)

    update_trainval_partitions(dataset, partition_key='trainval',required_partitions=required_partitions)
    update_trainval_partitions(dataset, partition_key='test',required_partitions=required_partitions)

    initial_partition_counts = {i: len(dataset.partition['trainval'][i]) for i in range(len(dataset.partition['trainval']))}
    initial_partition_images = {i: dataset.partition['trainval'][i].tolist() for i in range(len(dataset.partition['trainval']))}

    print("\n=== Initial State of Partitions ===")
    for i in initial_partition_counts:
        print(f"Partition[{i}]: {initial_partition_counts[i]} images")

    user_inputs = get_user_inputs()

    dataset_name = user_inputs["dataset_name"]
    attribute_name = user_inputs["attribute_name"]
    noise_level = user_inputs["noise_level"]

    default_image_prefix =f"{PREFIX_IMAGE}_{noise_level}_noise_"

    print(f"Using default image prefix: {default_image_prefix}")

    max_images_in_dataset = get_max_images(dataset_name, attribute_name)
    print(f"Max images for {attribute_name} in {dataset_name} : {max_images_in_dataset}")

    base_file_name = f"dataset_{dataset_name}_{noise_level}_{attribute_name}"

    postfix = input(f"The current prefix is '{base_file_name}'\n. Would you like to add anything to the file name? (Leave blank to skip): ").strip()
    if postfix:
        final_file_name = f"{base_file_name}_{postfix}.pkl"
    else:
        final_file_name = f"{base_file_name}.pkl"

    save_file_path = os.path.join(DATASET_PICKLE_PATH, final_file_name)

    added_images_list = []
    processed_images_count = 0

    moved_images = []
    already_present_images = []

    for root, dirs, files in os.walk(COMFYUI_OUTPUT):
        print("")
        print(f"Current prefix is: '{default_image_prefix}'")
        
        image_prefix = input(f"Enter a new prefix to filter or press Enter to continue with the current prefix: ").strip()

        if not image_prefix:
            image_prefix = default_image_prefix
        sorted_files = filter_images_by_prefix(sorted(files), image_prefix)

        if not sorted_files:
            print(f"No images found with the prefix '{image_prefix}' at {COMFYUI_OUTPUT}. Exiting.")
            sys.exit(1)

        required_images = (AUGMENTATION_PERCENTAGE_OPTIONS[-1] * max_images_in_dataset)
        if len(sorted_files) < required_images:
            print(f"Insufficient images for 200%. Required: {required_images}, Available: {len(sorted_files)}")
            print("Exiting program.")
            sys.exit(1)
        print("total files:",len(sorted_files))
        for file in sorted_files:
            if file.endswith('.png'):
                file_name_without_extension = os.path.splitext(file)[0]

                number = file_name_without_extension.split("_")[5]
                print("Number : ",number)
                txt_file = f"ComfyUI_{number}_.txt"

                desired_part = "_".join(file_name_without_extension.split("_")[:4])
                img_path = os.path.join(root, file)
                txt_path = os.path.join(root, txt_file)
                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        prompt = f.read().strip()
                    label_vector = get_labels_from_prompt(prompt)
                    added_images_list.append(img_path)
                    try:
                        is_moved = add_image_dataset(img_path, label_vector, dataset, labels_from_file,processed_images_count,max_images_in_dataset,unique_name=False)

                        if is_moved:
                            moved_images.append(img_path)
                        else:
                            already_present_images.append(img_path)
                        processed_images_count += 1
                        if processed_images_count >= max_images_in_dataset * AUGMENTATION_PERCENTAGE_OPTIONS[-1]:
                            print(f"Reached the limit of {max_images_in_dataset} images.")
                            break
                    except ValueError as e:
                        print(e)

        if processed_images_count >= max_images_in_dataset * (AUGMENTATION_PERCENTAGE_OPTIONS[-1]):
            break

    print("\n=== Summary of Dataset Update ===")
    print("\nImages Moved:")
    if moved_images:
        for img in moved_images:
            print(f" - {img}")
    else:
        print("No images were moved.")

    print("\nWarning: Images Already There:")
    if already_present_images:
        for img in already_present_images:
            print(f" - {img}")
    else:
        print("No duplicate images detected.")

    print(f"\nTotal Images Moved: {len(moved_images)}")
    print(f"Total Duplicate Images: {len(already_present_images)}")
    print("Images in total added in dataset file : ",len(moved_images)+len(already_present_images))

    save_dataset(dataset, save_file_path)

    final_partition_counts = {i: len(dataset.partition['trainval'][i]) for i in range(len(dataset.partition['trainval']))}
    final_partition_images = {i: dataset.partition['trainval'][i].tolist() for i in range(len(dataset.partition['trainval']))}

    print("\n=== Partition Updates ===")
    for i in final_partition_counts:
        added_count = final_partition_counts[i] - initial_partition_counts[i]
        added_images = set(final_partition_images[i]) - set(initial_partition_images[i])
        print(f"Partition[{i}]:")
        print(f"  Initial Count: {initial_partition_counts[i]} images")
        print(f"  Final Count: {final_partition_counts[i]} images")
        print(f"  Added: {added_count} images")
        if added_images:
            print(f"  Added Image Indices: {list(added_images)}")
        else:
            print(f"  No new images added.")

    print(f"Program completed. Dataset saved as: {save_file_path}")