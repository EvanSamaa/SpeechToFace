import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
import pickle
import random
from tqdm import tqdm
from transformers import Wav2Vec2Processor
from sklearn.model_selection import train_test_split
import librosa


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, subjects_dict, config, data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.config = config

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        if self.data_type == "train":
            subject = file_name.split("_")[0]
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels
        # audio, vertices, template, subject one-hot, file name
        if self.config["dataset"] == "meads":
            return torch.FloatTensor(audio), vertice, template, torch.FloatTensor(
            one_hot), file_name
        else:
            return torch.FloatTensor(audio), vertice, torch.FloatTensor(template), torch.FloatTensor(
            one_hot), file_name

    def __len__(self):
        return self.len


def read_data(config):
    print("Loading data...")
    data = defaultdict(dict)
    # data goes from key (file name) to 
    train_data = []
    valid_data = []
    test_data = []

    # audio and vertices (of output animation) path 
    audio_path = os.path.join(config["data_path"], config["dataset"], config["wav_path"])
    vertices_path = os.path.join(config["data_path"], config["dataset"], config["vertices_path"])
    
    # load audio pre-processor
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/hubert-xlarge-ls960-ft")  # HuBERT uses the processor of Wav2Vec 2.0

    # load template
    template_file = os.path.join(config["data_path"], config["dataset"], config["template_file"])
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    indices_to_split = []
    
    # subjects are represented by numbers 
    all_subjects = config["test_subjects"].split() + config["val_subjects"].split() + config["test_subjects"].split()
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                wav_path = os.path.join(r, f)
                key = f.replace("wav", "npy") # example of key: "3_0_06.wav" -> subject_id = 3, sentence_id = 6, emotion_id = 0
                if config["dataset"] == "meads":
                    # subject_id = key.split("_")[0]
                    # sentence_id = int(key.split(".")[0].split("_")[-1])
                    key = key.replace(".npy", ".pkl") 
                # get sample info from the name and add it to the dict for the splits
                if config["dataset"] == 'vocaset':
                    subject_id = "_".join(key.split("_")[:-1])
                    sentence_id = int(key.split(".")[0][-2:])
                else:
                    sentence_id = key.split(".")[0].split("_")[-1]
                    subject_id = key.split("_")[0]
                # skip subjects not included in the training or test sets for faster loading
                if subject_id not in all_subjects:
                    continue
                if config["dataset"] == 'beat' or config["dataset"] == 'meads':
                    # for beat dataset, we need to split the data into train, val, and test due to the additional emotion label
                    emotion_id = int(key.split(".")[0].split("_")[-2])
                    indices_to_split.append([sentence_id, emotion_id, subject_id])
                # load wav file at 16000 Hz
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                # process speech array with the pre-processor
                input_values = np.squeeze(processor(speech_array, return_tensors="pt", padding="longest",
                                         sampling_rate=sampling_rate).input_values)

                # store the audio, template, and vertices in the dict
                data[key]["audio"] = input_values
                temp = templates.get(subject_id, np.zeros(config["vertice_dim"]))
                data[key]["name"] = f
                if config["dataset"] == "meads":
                    data[key]["template"] = temp
                else:
                    data[key]["template"] = temp.reshape((-1)) # different file has a different template (VOCA, BIWI, etc.)
                vertice_path = os.path.join(vertices_path, f.replace("wav", "npy"))
                if config["dataset"] == "meads":
                    vertice_path = vertice_path.replace(".npy", ".pkl")
                if not os.path.exists(vertice_path):
                    del data[key]
                    print("Vertices Data Not Found! ", vertice_path)
                else:
                    data[key]["vertice"] = vertice_path

    # these are dicts where empty list are created for each key
    train_split = defaultdict(list)
    val_split = defaultdict(list)
    test_split = defaultdict(list)

    # for beat do a stratified split
    # it ensures a balanced representation of emotions across the sets

    if config["dataset"] == 'beat' : # the same speaker would appear in both the training, validation, and test sets
        # indices_to_split contains a list of [sentence_id, emotion_id, subject_id], each is a sample
        indices_to_split = np.array(indices_to_split)
        train_indices, test_indices = train_test_split(
            indices_to_split, test_size=0.1, stratify=indices_to_split[:, 1:3], random_state=42
        ) # train_indices is in the form of [sentence_id, emotion_id, subject_id], and is now independent from test indices
        train_indices, val_indices = train_test_split(
            train_indices, test_size=1 / 9, stratify=train_indices[:, 1:3], random_state=42
        ) # val_indices is a subset of train_indices is in the form of [sentence_id, emotion_id, subject_id], and is now independent from train indices

        # each input audio has a unique sentence id, so we only keep that
        for idx in train_indices:
            train_split[idx[-1]].append(int(idx[0])) # each subject has a list of sentence ids
        for idx in val_indices:
            val_split[idx[-1]].append(int(idx[0]))
        for idx in test_indices:
            test_split[idx[-1]].append(int(idx[0]))

    indices = list(range(1, 2538))
    random.Random(1).shuffle(indices)
    nr_samples = 100
    # the split only determines what sentences are in the training, validation, and test sets
    # the subjects are determined by the config file
    splits = {
        'BIWI': { # 40 sentences spoken by 14 person. train, val, test each has a subset of the sentences spoken by all people
            'train': range(1, 33), 
            'val': range(33, 37),
            'test': range(37, 41)
        },
        'multiface': { # multiface has 50 sentences.
            'train': list(range(1, 41)),
            'val': list(range(41, 46)),
            'test': list(range(46, 51))
        },
        'damm_rig_equal': {
            'train': indices[:int(0.8 * nr_samples)],
            'val': indices[int(0.8 * nr_samples):int(0.9 * nr_samples)],
            'test': indices[int(0.9 * nr_samples):nr_samples]
        },
        'beat': { # this has a lot more sentences. It is split to ensure balanced emotion subject representation
            'train': train_split,
            'val': val_split,
            'test': test_split
        },
        # overlapp?????? there's so much what 
        'vocaset': {'train': range(1, 41), 'val': range(21, 41), 'test': range(21, 41)}, 
        "meads": {
            "train": range(1, 24),
            "val": range(24, 27),
            "test": range(27, 31)
        }
    }


    subjects_dict = {}
    # list of subjects for each set
    subjects_dict["train"] = [i for i in config["train_subjects"].split(" ")]
    subjects_dict["val"] = [i for i in config["val_subjects"].split(" ")]
    subjects_dict["test"] = [i for i in config["test_subjects"].split(" ")]

    # use both the splits (randomly sampled sentences) and the subjects (subjects specified in config file) to determine the final datasets
    for k, v in data.items(): 
        if config["dataset"] == 'beat':
            subject_id = k.split("_")[0]
            sentence_id = int(k.split(".")[0].split("_")[-1])
            if subject_id in subjects_dict["train"] and sentence_id in splits[config["dataset"]]['train'][subject_id]:
                train_data.append(v)
            elif subject_id in subjects_dict["val"] and sentence_id in splits[config["dataset"]]['val'][subject_id]:
                valid_data.append(v)
            elif subject_id in subjects_dict["test"] and sentence_id in splits[config["dataset"]]['test'][subject_id]:
                test_data.append(v)
        elif config["dataset"] == 'meads':
            subject_id = k.split("_")[0]
            sentence_id = int(k.split(".")[0].split("_")[-1])
            if subject_id in subjects_dict["train"] and sentence_id in splits[config["dataset"]]['train']:
                train_data.append(v)
            elif subject_id in subjects_dict["val"] and sentence_id in splits[config["dataset"]]['val']:
                valid_data.append(v)
            elif subject_id in subjects_dict["test"] and sentence_id in splits[config["dataset"]]['test']:
                test_data.append(v)
        elif config["dataset"] == 'BIWI' or config["dataset"] == 'vocaset':
            subject_id = "_".join(k.split("_")[:-1])
            sentence_id = int(k.split(".")[0][-2:])
            if subject_id in subjects_dict["train"] and sentence_id in splits[config["dataset"]]['train']:
                train_data.append(v)
            elif subject_id in subjects_dict["val"] and sentence_id in splits[config["dataset"]]['val']:
                valid_data.append(v)
            elif subject_id in subjects_dict["test"] and sentence_id in splits[config["dataset"]]['test']:
                test_data.append(v)
        else:
            subject_id = k.split("_")[0]
            sentence_id = int(k.split(".")[0].split("_")[-1])
            if subject_id in subjects_dict["train"] and sentence_id in splits[config["dataset"]]['train']:
                train_data.append(v)
            elif subject_id in subjects_dict["val"] and sentence_id in splits[config["dataset"]]['val']:
                valid_data.append(v)
            elif subject_id in subjects_dict["test"] and sentence_id in splits[config["dataset"]]['test']:
                test_data.append(v)

    print(len(train_data), len(valid_data), len(test_data))
    # train, valid, test are each lists of dicts, where each dict contains the "audio", "template", and "vertices" of a sample
    return train_data, valid_data, test_data, subjects_dict


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(config):
    g = torch.Generator()
    g.manual_seed(0)
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(config)
    train_data = Dataset(train_data, subjects_dict, config, "train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True, worker_init_fn=seed_worker,
                                       generator=g)
    valid_data = Dataset(valid_data, subjects_dict, config, "val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    test_data = Dataset(test_data, subjects_dict, config, "test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return dataset


if __name__ == "__main__":
    get_dataloaders(config)

