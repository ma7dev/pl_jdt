import os, sys, copy
from tqdm.notebook import tqdm
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import os.path as osp


sys.path.insert(0, os.path.abspath('/home/mazen/Projects/pl_jdt/tmp/tracking_wo_bnw'))
from pl_jdt.obj_det.mot_data import MOTObjDetect
import pl_jdt.obj_det.transforms as T
from pl_jdt.obj_det.engine import train_one_epoch, evaluate
import pl_jdt.obj_det.utils as utils

DATA_DIR = '/home/mazen/Datasets/MOT17Det'
OUTPUT_DIR = "../../output/faster_rcnn_fpn/faster_rcnn_fpn_training_mot_17_split_09"

def get_detection_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.nms_thresh = 0.3
    
    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def evaluate_and_write_result_files(model, data_loader):
    print(f'EVAL {data_loader.dataset}')
    model.eval()
    iou_types = ["bbox"]
    coco_eval, results, loss_dicts = evaluate(model, data_loader, device, iou_types)
    evaluation_metrics = {'AP': coco_eval.coco_eval['bbox'].stats[0]}
    data_loader.dataset.write_results_files(results, OUTPUT_DIR)
    return evaluation_metrics, loss_dicts

def eval(model, data_loader, epoch=0, best_eval_metrics=None):
    print(f"VAL - Epoch {epoch}")
    print(f'VAL {data_loader.dataset}')
    evaluation_metrics, loss_dicts = evaluate_and_write_result_files(model, data_loader)
    for metric, metric_value in evaluation_metrics.items():
        print(f'VAL/{metric}', metric_value, 0)
    if len(loss_dicts) > 0:
        for loss_key in loss_dicts[0].keys():
            loss = torch.tensor([loss_dict[loss_key] for loss_dict in loss_dicts])
            print(f'VAL/{loss_key}', loss.mean(), 0)
    if best_eval_metrics is None:
        return copy.deepcopy(evaluation_metrics)
    else:
        for metric, metric_value in evaluation_metrics.items():
            if metric_value > best_eval_metrics[metric]:
                best_eval_metrics[metric] = metric_value

                print(f'Save best {metric} ({metric_value:.2f}) model at epoch: {epoch}')
                torch.save(model.state_dict(), osp.join(OUTPUT_DIR, f"best_{metric}.model"))

        return best_eval_metrics

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if not osp.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # train_split_seqs = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
    train_split_seqs = ['MOT17-09']
    test_split_seqs = ['MOT17-09']
    num_epochs = 30
    # for seq in test_split_seqs:
    #     train_split_seqs.remove(seq)

    dataset = MOTObjDetect(
        osp.join(DATA_DIR, 'train'),
        get_transform(train=True),
        split_seqs=train_split_seqs)

    dataset_no_random = MOTObjDetect(
        osp.join(DATA_DIR, 'train'),
        get_transform(train=False),
        split_seqs=train_split_seqs)

    dataset_test = MOTObjDetect(
        osp.join(DATA_DIR, 'train'),
        get_transform(train=False),
        split_seqs=test_split_seqs)
    torch.manual_seed(1)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    data_loader_no_random = torch.utils.data.DataLoader(
        dataset_no_random, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    model = get_detection_model(dataset.num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.00001,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=10,
                                                gamma=0.1)
    best_eval_metrics = eval(model, data_loader_test)

    for epoch in range(1, num_epochs + 1):
        print(f"TRAIN - Epoch {epoch}/{num_epochs}")
        print(f'TRAIN {data_loader.dataset}')
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=200)
        
        # update the learning rate
        lr_scheduler.step()
        
        # evaluate on the test dataset
        if epoch % 2 == 0:
            best_eval_metrics = eval(model, data_loader_test, epoch, best_eval_metrics)
