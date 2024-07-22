import torch

class MIBCI2aDataset(torch.utils.data.Dataset):
    def _getFeatures(self, filePath):
        # implement the getFeatures method
        """
        read all the preprocessed data from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        import numpy as np
        import os
        npy_files = [f for f in os.listdir(filePath)]
        data = []
        for fileName in npy_files:
            data.append(np.load(filePath+fileName))
        features=np.concatenate(data,axis=0)
        features=torch.Tensor(features).unsqueeze(1)
        return features
        pass

    def _getLabels(self, filePath):
        # implement the getLabels method
        """
        read all the preprocessed labels from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        import numpy as np
        import os
        npy_files = [f for f in os.listdir(filePath)]
        data = []
        for fileName in npy_files:
            data.append(np.load(filePath+fileName))
        labels=np.concatenate(data,axis=0)
        labels=torch.Tensor(labels).to(torch.long)
        return labels
        pass

    def __init__(self, mode, model,batch_size=288):
        # remember to change the file path according to different experiments
        assert mode in ['train', 'test', 'finetune']
        if model=="SD":
            if mode == 'train':
                # subject dependent: ./dataset/SD_train/features/ and ./dataset/SD_train/labels/
                # leave-one-subject-out: ./dataset/LOSO_train/features/ and ./dataset/LOSO_train/labels/
                self.features = self._getFeatures(filePath='./dataset/SD_train/features/')
                self.labels = self._getLabels(filePath='./dataset/SD_train/labels/')
                self.batch_num=int(self.labels.shape[0]/batch_size)
                f=[0]*self.batch_num
                l=[0]*self.batch_num
                cnt=0
                for i in range(self.batch_num):
                    f[i]=self.features[cnt:cnt+batch_size]
                    l[i]=self.labels[cnt:cnt+batch_size]
                    cnt+=batch_size
                self.features = f
                self.labels = l

            if mode == 'finetune':
                # finetune: ./dataset/FT/features/ and ./dataset/FT/labels/
                self.features = self._getFeatures(filePath='./dataset/FT/features/')
                self.labels = self._getLabels(filePath='./dataset/FT/labels/')
            if mode == 'test':
                # subject dependent: ./dataset/SD_test/features/ and ./dataset/SD_test/labels/
                # leave-one-subject-out and finetune: ./dataset/LOSO_test/features/ and ./dataset/LOSO_test/labels/
                self.features = self._getFeatures(filePath='./dataset/SD_test/features/')
                self.labels = self._getLabels(filePath='./dataset/SD_test/labels/')
        else:
            if mode == 'train':
                # subject dependent: ./dataset/SD_train/features/ and ./dataset/SD_train/labels/
                # leave-one-subject-out: ./dataset/LOSO_train/features/ and ./dataset/LOSO_train/labels/
                self.features = self._getFeatures(filePath='./dataset/LOSO_train/features/')
                self.labels = self._getLabels(filePath='./dataset/LOSO_train/labels/')
                self.batch_num=int(self.labels.shape[0]/batch_size)
                f=[0]*self.batch_num
                l=[0]*self.batch_num
                cnt=0
                for i in range(self.batch_num):
                    f[i]=self.features[cnt:cnt+batch_size]
                    l[i]=self.labels[cnt:cnt+batch_size]
                    cnt+=batch_size
                self.features = f
                self.labels = l
            if mode == 'finetune':
                # finetune: ./dataset/FT/features/ and ./dataset/FT/labels/
                self.features = self._getFeatures(filePath='./dataset/FT/features/')
                self.labels = self._getLabels(filePath='./dataset/FT/labels/')
            if mode == 'test':
                # subject dependent: ./dataset/SD_test/features/ and ./dataset/SD_test/labels/
                # leave-one-subject-out and finetune: ./dataset/LOSO_test/features/ and ./dataset/LOSO_test/labels/
                self.features = self._getFeatures(filePath='./dataset/LOSO_test/features/')
                self.labels = self._getLabels(filePath='./dataset/LOSO_test/labels/')

    def __len__(self):
        # implement the len method
        return self.labels.shape[0]
        pass

    def __getitem__(self, idx):
        # implement the getitem method
        return (self.features[idx,:,:],self.labels[idx])
        pass