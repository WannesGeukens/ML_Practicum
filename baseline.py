# ---------- imports ---------- #
import sklearn
import csv

from dabstract.dataset import *
from dabstract.dataprocessor.processing_chain import *
from dabstract.dataprocessor.processors.processors import *
from dabstract.utils import get_class

from core.dataset import DCASE2020Task1b_MLlab

# ---------- inits ---------- #
# -- save strings
feature_name = 'baseline' # (do not change)
model_name = 'students'
mode = 'test' # 'test' or 'kaggle'

# -- set dirs (do not change)
results_path = os.path.join('results', model_name, feature_name, mode)
# Use following paths if you want to work on the SERVER (use remote Python interpreter of VENUS)
#paths = {'data': '/media/SoundData1/ML_lab_2020/DCASE2020/datasets/TAU-urban-acoustic-scenes-2020-3class-development/audio',
#         'meta': '/media/SoundData1/ML_lab_2020/DCASE2020/datasets/TAU-urban-acoustic-scenes-2020-3class-development',
#         'feat': '/media/SoundData1/ML_lab_2020/DCASE2020/features'}
# Use following paths if you want to work on your own PC (use local Python interpreter folder)
#paths = {'data': os.path.join('data','DCASE2020','datasets','TAU-urban-acoustic-scenes-2020-3class-development','audio'),
#         'meta': os.path.join('data','DCASE2020','datasets','TAU-urban-acoustic-scenes-2020-3class-development'),
#         'feat': os.path.join('data','DCASE2020','features')}
# Use following paths if you want to work on your own PC (use external folder)
paths = {'data': 'D:\ML_data\DCASE2020\datasets\TAU-urban-acoustic-scenes-2020-3class-development\\audio',
         'meta': 'D:\ML_data\DCASE2020\datasets\TAU-urban-acoustic-scenes-2020-3class-development',
         'feat': 'D:\ML_data\DCASE2020\\features'}


# -- flow params (do not change)
workers = 5
load_memory = False
verbose = True
do_train, overwrite_train = True, True
do_eval = True

# -- Get dataset
data = DCASE2020Task1b_MLlab(paths=paths, mode=mode)

# -- Get feature extraction
fe = get_class(feature_name,'core.feature_extraction')
pprint(fe.summary())

# -- Extract features
data.prepare_feat('audio',
                  feature_name,
                  fe,
                  verbose=verbose,
                  workers=workers,
                  new_key='input')

# -- load model
model = get_class(model_name, 'core.model')

# -- Do additional feature extraction on the fly
data.add_map('input', model.extract_features)

# -- load memory
if load_memory:
    if verbose: print("Loading data in memory...")
    data.load_memory('input',
                     workers=workers,
                     verbose=verbose)

# -- summary
if verbose: print('\n # --- Dataset summary --- #')
pprint(data.summary())

# --- on the fly processing on input
if verbose: print('\n # --- On-the-fly processing chains --- #')
processor = processing_chain().add(Normalizer(type='standard'))

# --- Get metric
metric = (lambda ref,est: sklearn.metrics.f1_score(ref,est,average='macro'))

# ---------- experiment loop ---------- #
score = []
for fold in range(data.xval['folds']):
    # --- Training
    savefile_model = os.path.join(results_path, "model_fold" + str(fold))
    if (do_train & (not model.exists(savefile_model))) | overwrite_train:
        if verbose: print('\n # --- Training fold ' + str(fold) + ' --- #')
        # --- create dir
        os.makedirs(results_path, exist_ok=True)

        # --- init processing chain
        savefile_processor = os.path.join(results_path, "processor_fold" + str(fold))  # save directory
        if (not os.path.isfile(savefile_processor + ".pickle")) or overwrite_train:
            processor.fit(data=data.get_xval_set(set='train', fold=fold)['input'])
            with open(savefile_processor + ".pickle", 'wb') as f:
                pickle.dump(processor, f)  # save
        else:
            with open(savefile_processor + ".pickle", "rb") as f:
                processor = pickle.load(f)  # load

        # -- get dataloaders
        train_data = DataAbstract(data.get_xval_set(fold=fold, set='train'), workers=workers)
        val_data = DataAbstract(data.get_xval_set(fold=fold, set='val'), workers=workers)

        # --- learn/validate model using
        model.train(x = train_data['input'][:],
                    y = train_data['target'][:],
                    x_val = val_data['input'][:],
                    y_val = val_data['target'][:],
                    metric_fct = metric)
        # --- save model
        model.save(savefile_model)

    # ---------- Evaluation ---------- #
    if do_eval:
        if verbose: print('\n # --- Evaluating fold ' + str(fold) + ' --- #')
        # --- Load processing chain
        savefile_processor = os.path.join(results_path, "processor_fold" + str(fold))  # save directory
        with open(savefile_processor + ".pickle", "rb") as f:
            processor = pickle.load(f)  # load

        # -- get dataloader
        test_data = DataAbstract(data.get_xval_set(fold=fold,set='test'), workers=workers)

        # -- load model parameters
        model.load(savefile_model)

        # --- predict
        estimate = model.predict(test_data['input'][:])

        # -- Evaluate
        if mode=='test':
            # get score
            score.append(metric(test_data['target'][:], estimate))
            # print
            print('Fold ' + str(fold) + ' with a F1-score of ' + str(score[-1]))
        elif mode=='kaggle':
            # save estimates to kaggle csv format
            with open(os.path.join(results_path, model_name) + ".csv", 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Id', 'Label'])
                for k in range(0, len(estimate)):
                    writer.writerow([k+1, np.int(estimate[k])])

# ---------- Evaluation average results ---------- #
if mode == 'test':
    print('Average F1 score is ' + str(np.mean(score)) + ' with a std of ' + str(np.std(score)))