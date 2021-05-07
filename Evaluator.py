import sys

import numpy as np
from scipy.stats import mode
from matplotlib import pyplot as plt
from skimage.transform import resize
import tensorflow as tf

from Model import Model
from TileDataGenerator import TileDataGenerator
from FullImageDataGenerator import FullImageDataGenerator

class Evaluator:

    @classmethod
    def evaluate(cls, model, data, train_val, overlap='minimum', min_area=4000):
        """Compute evaluation metrics (detection precision/recall + segmentation MCC) on each image, annotation pair.

        First, the model is used to get the prediction mask. Postprocessing is then applied, before 
        computing the metrics.
        data_x, data_y should be numpy arrays with the samples in the first axis, or lists of numpy arrays.
        """
        metrics = []

        if train_val == 'train':
            data_x = [data.full_images[idx] for idx in data.train_idxs]
            data_y = [data.full_annotations[idx] for idx in data.train_idxs]
        elif train_val == 'val':
            data_x = [data.full_images[idx] for idx in data.val_idxs]
            data_y = [data.full_annotations[idx] for idx in data.val_idxs]
        else:
            raise ValueError('train_val must be equal to train or val')

        for im, anno in zip(data_x, data_y):
            if( isinstance(data, TileDataGenerator) ):
                tiles = data._get_regular_tiling(im.shape, overlap)
                tiles_x = []
                tiles_y = []
                for tx,ty in tiles:
                    tiles_x += [im[ty:ty+data.tile_size[0],tx:tx+data.tile_size[1]]]
                    tiles_y += [anno[ty:ty+data.tile_size[0],tx:tx+data.tile_size[1]]]
                tiles_prediction = model.predict(np.array(tiles_x))
                
                pred_image = data.stitch(tiles_prediction, anno.shape, overlap)
                pred_labels = model.post_process(pred_image, min_area)
            else:
                im_ = data.preprocess_image(im)
                pred_image = resize(model.predict(np.array([im_]))[0], im.shape[:2])
                pred_labels = model.post_process(pred_image, min_area)
            
            metrics += [cls._get_metrics(anno, pred_labels)]
        return np.array(metrics)


    @staticmethod
    def _get_metrics(gt_labels, pred_labels):
        """Compute evaluation metrics (detection precision/recall + segmentation MCC) on a 
        pair of ground truth labels / predicted labels.
        """   
        # Remove regions which are < 5 pixels in trueLabels (sometimes there are 1-2 isolated pixels in the corner 
        # -> not really fair to include them)
        for i in range(1, gt_labels.max()+1):
            if( (gt_labels==i).sum() < 5 ):
                gt_labels[gt_labels==i] = 0

        trueLabels = np.unique(gt_labels)
        trueLabels = trueLabels[trueLabels>0].astype('int')
        predLabels = np.unique(pred_labels)
        predLabels = predLabels[predLabels>0].astype('int')


        best_matches = np.zeros((len(predLabels),3)) # predLabel, gtLabel, isValidMatch
        best_matches[:,0] = predLabels
        for i in range(len(predLabels)):
            predObject = pred_labels==predLabels[i] # select predicted object
            corrRegionInGT = gt_labels[predObject]  # find region in gt image
            if corrRegionInGT.max() > 0: # if it's only background, there's no match
                bestMatch = mode(corrRegionInGT[corrRegionInGT>0])[0][0]  # mode of the region = object with largest overlap
                matchInGT = gt_labels==bestMatch    # Select GT object 
                best_matches[i,1] = bestMatch       
                overlap = predObject*matchInGT      # Select overlapping region
                best_matches[i,2] = (overlap.sum()/matchInGT.sum())>0.5 # if #overlapping pixels > 50% GT object pixels : valid
        
        TP = int(best_matches[:,2].sum())
        FP = int((best_matches[:,2]==0).sum())
        FN = int(len(trueLabels)-TP)
        
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        
        gt_mask = gt_labels>0
        pred_mask = pred_labels>0
        TP = float(((gt_mask==True)*(pred_mask==True)).sum())
        FP = float(((gt_mask==False)*(pred_mask==True)).sum())
        FN = float(((gt_mask==True)*(pred_mask==False)).sum())
        TN = float(((gt_mask==False)*(pred_mask==False)).sum())
        MCC = ((TP*TN)-(FP*FN))/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        
        return precision, recall, MCC

def main():
    try:
        path_to_dataset = sys.argv[1]
        path_to_model = sys.argv[2]
        tile = sys.argv[3]=='tile'
    except:
        print("path_to_dataset, path_to_model & tile must be provided.")
        sys.exit(1)

    tf.keras.backend.clear_session()

    dataset = TileDataGenerator(5, 10, path_to_dataset, (256,384)) if tile else FullImageDataGenerator(5, 10, path_to_dataset, (256,384))
    model = Model((256,384),path_to_model,loadFrom=path_to_model)
    
    train_metrics = Evaluator.evaluate(model, dataset, 'train')
    val_metrics = Evaluator.evaluate(model, dataset, 'val')

    with open(f"{path_to_model}_metrics.txt", 'w') as fp:
        print("Training perfomance:", file=fp)
        print("Precision\tRecall\tMCC", file=fp)
        print(train_metrics.mean(axis=0), file=fp)
        print(np.median(train_metrics,axis=0), file=fp)
        print(" ---- ", file=fp)
        print("Validation perfomance:", file=fp)
        print("Precision\tRecall\tMCC", file=fp)
        print(val_metrics.mean(axis=0), file=fp)
        print(np.median(val_metrics,axis=0), file=fp)

    np.save(f"{path_to_model}_metrics_train.npy", train_metrics)
    np.save(f"{path_to_model}_metrics_val.npy", val_metrics)

if __name__ == '__main__':
    main()