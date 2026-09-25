import shutil
import os
import time
import argparse
import logging
import sys
import pdb
import glob
import pickle
import csv

import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

from torch.utils.tensorboard import SummaryWriter


from data.dataloader import train_ds, val_ds, test_ds_dict
from run.utils import Aggremeter, write_metrix, evaluate_prediction
import model.model as model
import warnings
warnings.simplefilter("ignore")

from run.Args import args

Net = model.Multi_view_Knee
# Net = model.Pretrain_Encoder

@torch.no_grad()
def compute_epoch_difficulty_weights(net, ds, args):
    net.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    difficulties = []
    for images, label, clinic, report, prompt, _ in tqdm(loader, desc='Computing difficulty', leave=False):
        if torch.cuda.is_available():
            images = [image.to(device) for image in images]
            clinic = clinic.to(device)
            report = report.to(device)
            prompt = prompt.to(device)
        logits = net(images, clinic, report, prompt)  # (final_pred, mm_pred, mv_pred)
        final_logits = logits[0]  # [b, 8]
        task_logits = torch.chunk(final_logits, chunks=2, dim=1)  # list of 2 x [b,4]
        ent_list = []
        for tl in task_logits:
            probs = torch.softmax(tl, dim=1)
            logp = torch.log(probs + 1e-8)
            ent = -(probs * logp).sum(dim=1)  # [b]
            ent_norm = ent / torch.log(torch.tensor(probs.shape[1], dtype=probs.dtype, device=probs.device))
            ent_list.append(ent_norm)
        ent_avg = torch.stack(ent_list, dim=0).mean(dim=0)  # [b]
        difficulties.append(ent_avg.detach().cpu())
    if len(difficulties) == 0:
        return None
    ent_all = torch.cat(difficulties, dim=0)  # [N]
    beta = getattr(args, 'entropy_beta', 0.1)
    weights = torch.exp(beta * ent_all)
    weights = weights / (weights.mean() + 1e-8)
    return weights.tolist()

def train_model(model, train_loader, epoch, optimizer, act_task, scalar, args):
    model.train()

    agg_meter = Aggremeter()
    tbar = tqdm(train_loader)
    final = True
    for i, (images, label, clinic, report, prompt, _) in enumerate(tbar):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            images = [image.cuda() for image in images]
            label = label.cuda()
            clinic = clinic.cuda()
            report = report.cuda()
            prompt = prompt.cuda()


        if args.half:
            with autocast(device_type='cuda', dtype=torch.float16):
                logits= model(images, clinic, report, prompt) # prediction= (final_pred, sag_pred, cor_pred, axi_pred)
                loss, loss_value = model.criterion(logits, label, act_task, final)
            scalar.scale(loss).backward()

            if (i+1) % args.iters_to_accumulate == 0 or (i+1) == len(train_loader):
                scalar.step(optimizer)
                scalar.update()

            logits = [x.type(torch.FloatTensor) for x in logits]

        else:
            logits= model(images, clinic, report, prompt) # prediction= (final_pred, sag_pred, cor_pred, axi_pred)
            loss, loss_value = model.criterion(logits, label, act_task, final)
            loss.backward()
            optimizer.step()
            # if (i+1) % args.iters_to_accumulate == 0 or (i+1) == len(train_loader):
            #     optimizer.step()
            #     optimizer.zero_grad()

        prediction = [x.tolist() for x in logits]
        gt = label.tolist()

        agg_meter.add_pred(prediction[0], gt)
        agg_meter.add_loss(loss_value)

        if args.debug and i == 2:
            break

    auc_dict, acc_dict, metrix_dict = evaluate_prediction(agg_meter.pred_list, agg_meter.label_list, metrix_output=True)
    if args.write_metrix:
        write_metrix(metrix_dict, args.log_root_folder, "train_final", args)

    agg_meter.add_metrix(list(acc_dict.values()), list(auc_dict.values()))
    train_loss_epoch = agg_meter.loss
    train_auc_epoch = agg_meter.auc
    train_acc_epoch = agg_meter.acc
    return train_loss_epoch, train_auc_epoch, train_acc_epoch

def evaluate_model(model, val_loader, epoch, args, mode = 'Valid', save_path=None, save_results=False):
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    agg_meter = Aggremeter()
    mm_agg_meter = Aggremeter()
    mv_agg_meter = Aggremeter()

    tbar = tqdm(val_loader)
    for i, (images, label, clinic, report, prompt, id) in enumerate(tbar):
        
        
        if torch.cuda.is_available():
            images = [image.cuda() for image in images]
            label = label.cuda()
            clinic = clinic.cuda()
            report = report.cuda()
            prompt = prompt.cuda()


        logits = model(images, clinic, report, prompt) # prediction= (final_pred, sag_pred, cor_pred, axi_pred)
        _, loss_value = model.criterion(logits, label)
        
        prediction = [x.tolist() for x in logits]
        gt = label.tolist()
        agg_meter.add_pred(prediction[0], gt)
        agg_meter.add_loss(loss_value)
        mm_agg_meter.add_pred(prediction[1], gt)
        mv_agg_meter.add_pred(prediction[2], gt)

        if args.debug:
            break

    _, _, mm_metrix_dict = evaluate_prediction(mm_agg_meter.pred_list, mm_agg_meter.label_list, metrix_output=True)
    _, _, mv_metrix_dict = evaluate_prediction(mv_agg_meter.pred_list, mv_agg_meter.label_list, metrix_output=True)
    final_auc, final_acc, fin_metrix_dict = evaluate_prediction(agg_meter.pred_list, agg_meter.label_list, metrix_output=True)

    agg_meter.add_metrix(list(final_acc.values()), list(final_auc.values()) )
    val_loss_epoch = agg_meter.loss
    val_auc_epoch = agg_meter.auc
    val_acc_epoch = agg_meter.acc

    if save_results:
        with open(os.path.join(save_path, 'results%s.pk'%mode),'wb+') as outfile:
            pickle.dump(agg_meter, outfile)
        if args.write_metrix:
            write_metrix([fin_metrix_dict, mm_metrix_dict, mv_metrix_dict], save_path, [mode + "final", "mmb", "mvb"], args)

    return val_loss_epoch, val_auc_epoch, val_acc_epoch

def run():
    net = Net(backbone=args.backbone, encoder_layer=args.model_depth, pretrain=True, kargs=args)

    log_root_folder = args.log_folder

    setattr(args, "log_root_folder", log_root_folder)
    MODELSPATH = log_root_folder
    if args.flush_history == 1:
        objects = os.listdir(log_root_folder)
        for f in objects:
            if os.path.isdir(log_root_folder + f):
                shutil.rmtree(log_root_folder + f)
    os.makedirs(log_root_folder)

    script_save_folder = os.path.join(log_root_folder, 'scripts')
    os.mkdir(script_save_folder)
    for script in glob.glob("*.py"):
        dst_file = os.path.join(log_root_folder, 'scripts', os.path.basename(script))
        shutil.copyfile(script, dst_file)

    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )
    fh = logging.FileHandler(os.path.join(log_root_folder, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    
    # net = torch.compile(net)

    
    if torch.cuda.is_available():
        net = net.cuda()

    # optimizer = optim.SGD(params=net.parameters(), lr = args.lr, weight_decay=0.1)
    optimizer = torch.optim.Adam(params=net.parameters(), lr = args.lr)
    scheduler = ExponentialLR(optimizer=optimizer, gamma=0.9, verbose=True)

    best_val_loss = float('inf')
    best_val_auc = float(0)
    best_model_file = None

    num_epochs = args.epochs
    iteration_change_loss = 0
    patience = args.patience
    log_every = args.log_every

    t_start_training = time.time()
    
    act_task = -1 # which class to train

    for epoch in range(num_epochs):

        t_start = time.time()
        
        train_ds.balance_cls(act_task)

        use_epoch_weighting = True
        if use_epoch_weighting:
            weights = compute_epoch_difficulty_weights(net, train_ds, args)
        else:
            weights = None

        if weights is not None:
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_ds), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, sampler=sampler, shuffle=False)
        else:
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        validation_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        test_loader = DataLoader(test_ds_dict['Internal'], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

        scalar = GradScaler()

        if args.data_balance:
            if act_task == 11:
                act_task = -1
            else:
                act_task = act_task+1

        train_loss, train_auc, train_acc = train_model(
            net, train_loader, epoch, optimizer, act_task, scalar, args)

        if epoch % 10 == 0:
            scheduler.step()
        
        with torch.no_grad():
            val_loss, val_auc, val_acc = evaluate_model(
                net, validation_loader, epoch, args, save_path=log_root_folder, save_results=False)

            test_loss, test_auc, test_acc = evaluate_model(
                net, test_loader, epoch, args, save_path=log_root_folder, mode='Test', save_results=False)


        t_end = time.time()
        delta = t_end - t_start

        logging.info(
            "Epoch {10}\n train loss {0} | train auc {1} | train acc {2} |\n val loss {3} | val auc {4} | val acc {5} |\n test loss {6} | test auc {7} | test acc {8} elapsed time {9} s".format(
                train_loss, train_auc, train_acc, val_loss, val_auc, val_acc, test_loss, test_auc, test_acc, delta, epoch+1))

        iteration_change_loss += 1
        logging.info('-' * 50)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            iteration_change_loss = 0
            
            if best_model_file is not None and os.path.exists(f'{log_root_folder}/{best_model_file}'):
                os.remove(f'{log_root_folder}/{best_model_file}')
                logging.info(f'Removed previous best model: {best_model_file}')
            
            net.save_or_load_encoder_para(path=log_root_folder)
            
            if bool(args.save_model):
                file_name = f'model_{val_auc:0.4f}_train_auc_{train_auc:0.4f}_test_auc_{test_auc:0.4f}_epoch_{epoch + 1}_bestval.pth'
                try:
                    exported_model = net
                    torch.save(exported_model.state_dict(), f'{log_root_folder}/{file_name}')
                    best_model_file = file_name
                    logging.info(f'Saved new best model: {file_name}')
                except:
                    pass
            
            with torch.no_grad():
                val_loss, val_auc, val_acc = evaluate_model(
                    net, validation_loader, epoch, args, save_path=log_root_folder, save_results=True)
                test_loss, test_auc, test_acc = evaluate_model(
                    net, test_loader, epoch, args, save_path=log_root_folder, mode='Test', save_results=True)
            logging.info('Saved resultsValid.pk and resultsTest.pk for best validation loss')

        if iteration_change_loss == patience:
            logging.info('Early stopping after {0} iterations without the decrease of the val loss'.
                  format(iteration_change_loss))
            break

    t_end_training = time.time()
    logging.info(f'training took {t_end_training - t_start_training} s')

