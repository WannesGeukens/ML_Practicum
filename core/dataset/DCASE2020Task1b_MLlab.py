import dcase_util
import pandas
import copy
import csv

from dabstract.dataprocessor.processing_chain import processing_chain
from dabstract.dataset.dataset import dataset
from dabstract.dataset.xval import *
from dabstract.dataprocessor.processors import *
from dabstract.utils import stringlist2ind

class DCASE2020Task1b_MLlab(dataset):
    def __init__(self,
                 paths=None,
                 split=None,
                 filter=None,
                 test_only=0,
                 mode='xval',
                 **kwargs):
        # other init
        self.mode = mode
        # init dict abstract
        super().__init__(name=self.__class__.__name__,
                         paths=paths,
                         split=split,
                         filter=filter,
                         test_only=test_only)
        # init mode
        self.init_mode()

    # Data: get data
    def set_data(self, paths):
        # -- Set Data
        # audio
        chain = processing_chain().add(WavDatareader(select_channel=0))
        self.add_subdict_from_folder('audio',paths['data'],map_fct=chain,save_path=os.path.join(paths['feat'],self.__class__.__name__, 'audio', 'raw'))
        # get meta
        labels = pandas.read_csv(os.path.join(paths['meta'], 'meta.csv'), delimiter='\t')
        # make sure audio and meta is aligned
        filenames = labels['filename'].to_list()
#        resort = np.array([filenames.index(os.path.join('audio',filename)) for filename in self['audio']['example']])
        resort = np.array([filenames.index('audio/' + filename ) for filename in self['audio']['example']])
        labels = labels.reindex(resort)
        # add labels
        self.add('identifier', labels['identifier'].to_list())
        self.add('source', labels['source_label'].to_list())
        self.add('scene', labels['scene_label'].to_list())
        self.add('scene_id', stringlist2ind(labels['scene_label'].to_list()))
        self.add('group', stringlist2ind(labels['identifier'].to_list()))
        self.add('example_id', np.arange(0, len(self)))
        self.add_alias('scene_id', 'target')
        return self

    def init_mode(self):
        # -- Split in two datasets
        # create indices
        np.random.seed(seed=0)
        shuffleInd = np.arange(len(self))
        np.random.shuffle(shuffleInd)
        testsetind = shuffleInd[0:int(0.7*len(self))]
        kagglesetind = shuffleInd[int(0.7*len(self)):]
        # Write GT CSV file for kaggle
        write = True
        if write:
            with open("solution.csv", 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Id', 'Label', 'Usage'])
                for k in range(0, len(kagglesetind)):
                    if (k % 2) == 0:
                        writer.writerow([k + 1, np.int(self['scene_id'][kagglesetind[k]]), 'Private'])
                    else:
                        writer.writerow([k + 1, np.int(self['scene_id'][kagglesetind[k]]), 'Public'])
        # remove labels of testset
        for k in kagglesetind:
            self['scene_id'][k] = -1
            self['scene'][k] = 'unknown'
            self['test_only'][k] = (-1 if self.mode=='test' else 1) #do not use (-1) or test set only (1)
        # create datasets
        testset, kaggleset = copy.deepcopy(self), copy.deepcopy(self)
        testset.add_select(testsetind)
        kaggleset.add_select(kagglesetind)
        # merge again
        self._data = testset._data + kaggleset._data
        # -- Set train val test sets
        np.random.seed(seed=0)
        if self.mode=='test':
            self.set_xval(group_random_kfold(folds=4, val_frac=1/3))
        elif self.mode=='kaggle':
            self.set_xval(group_random_split(val_frac=1/3, test_frac=0))
        else:
            raise NotImplementedError('Please use test or kaggle for the mode.')

    def prepare(self,paths):
        dcase_util.datasets.dataset_factory(
            dataset_class_name='TAUUrbanAcousticScenes_2020_3Class_DevelopmentSet',
            data_path=os.path.split(os.path.split(paths['data'])[0])[0],
        ).initialize()