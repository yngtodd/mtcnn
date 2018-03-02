import numpy as np
from torch.utils.data.dataset import Dataset


class Deidentified(Dataset):
    def __init__(self, data_path, label_path, transform=None):
        """
        Deidentified data.
        
        Parameters:
        ----------
        * `data_path`: [str]
            Path to the sentence data.
        
        * `label_path`: [str]
            Path to the data labels.
        """
        self.sentences = np.load(data_path + '/data.npy')
        self.subsite = np.load(label_path + '/subsite.npy')
        self.laterality = np.load(label_path + '/laterality.npy')
        self.behavior = np.load(label_path + '/behavior.npy')
        self.grade = np.load(label_path + '/grade.npy')
        self.transform = transform
    
    def __len__(self):
        return len(self.sentences)
        
    def __getitem__(self, idx):
        """
        Returns a sample of the data.
        """
        sentence = self.sentences[idx]
        subsite = self.subsite[idx]
        laterality = self.laterality[idx]
        behavior = self.behavior[idx]
        grade = self.grade[idx]
        
        sample = {
            'sentence': sentence,
            'subsite': subsite,
            'laterality': laterality,
            'behavior': behavior,
            'grade': grade
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
