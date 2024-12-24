import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net

        self.loss_xbd = config.loss_xbd
        self.loss_mmd = config.loss_mmd

        self.xbd_metrics_train = Evaluator(num_class=config.num_classes)
        self.xbd_metrics_val = Evaluator(num_class=config.num_classes)
        self.teq_metrics_train = Evaluator(num_class=config.num_classes)
        self.teq_metrics_val = Evaluator(num_class=config.num_classes)

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        # only net is used in the prediction/inference
        seg_pre = self.net(x)
        return seg_pre

    def get_avg_loss(self, step_outputs):
        loss_names = step_outputs[0].keys()
        loss_len = len(step_outputs)
        loss = {}
        for ln in loss_names:
            loss[ln] = (
                sum([o[ln].cpu().detach().numpy() for o in step_outputs]) / loss_len
            )

        return loss

    def training_step(self, batch, batch_idx):
        xbd_img, xbd_mask, teq_img, teq_mask = (
            batch["xbd_img"],
            batch["xbd_gt_semantic_seg"],
            batch["teq_img"],
            batch["teq_gt_semantic_seg"],
        )

        xbd_output = self.net(xbd_img)
        teq_output = self.net(teq_img)
        loss_xbd = self.loss_xbd(
            (xbd_output["logits"], xbd_output["logits_aux"]), xbd_mask
        )
        loss_mmd = self.loss_mmd(teq_output["embeddings"], xbd_output["embeddings"])

        loss = loss_xbd + loss_mmd

        if self.config.use_aux_loss:
            xbd_pre_mask = nn.Softmax(dim=1)(xbd_output["logits"])
            teq_pre_mask = nn.Softmax(dim=1)(teq_output["logits"])
        else:
            xbd_pre_mask = nn.Softmax(dim=1)(xbd_output["logits"])
            teq_pre_mask = nn.Softmax(dim=1)(teq_output["logits"])

        xbd_pre_mask = xbd_pre_mask.argmax(dim=1)
        teq_pre_mask = teq_pre_mask.argmax(dim=1)
        for i in range(xbd_mask.shape[0]):
            self.xbd_metrics_train.add_batch(
                xbd_mask[i].cpu().numpy(), xbd_pre_mask[i].cpu().numpy()
            )
            self.teq_metrics_train.add_batch(
                teq_mask[i].cpu().numpy(), teq_pre_mask[i].cpu().numpy()
            )
        out = {"loss": loss, "loss_xbd": loss_xbd, "loss_mmd": loss_mmd}
        self.training_step_outputs.append(out)
        return out

    def on_train_epoch_end(self):
        if "vaihingen" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif "potsdam" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif "whubuilding" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif "massbuilding" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif "cropland" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            xbd_mIoU = np.nanmean(self.xbd_metrics_train.Intersection_over_Union())
            xbd_F1 = np.nanmean(self.xbd_metrics_train.F1())
            teq_mIoU = np.nanmean(self.teq_metrics_train.Intersection_over_Union())
            teq_F1 = np.nanmean(self.teq_metrics_train.F1())

        xbd_OA = np.nanmean(self.xbd_metrics_train.OA())
        xbd_iou_per_class = self.xbd_metrics_train.Intersection_over_Union()
        teq_OA = np.nanmean(self.teq_metrics_train.OA())
        teq_iou_per_class = self.teq_metrics_train.Intersection_over_Union()
        eval_value = {
            "xbd_mIoU": xbd_mIoU,
            "xbd_F1": xbd_F1,
            "xbd_OA": xbd_OA,
            "teq_mIoU": teq_mIoU,
            "teq_F1": teq_F1,
            "teq_OA": teq_OA,
        }
        print("train:", eval_value)

        xbd_iou_value = {}
        teq_iou_value = {}
        for class_name, xbd_iou, teq_iou in zip(
            self.config.classes, xbd_iou_per_class, teq_iou_per_class
        ):
            xbd_iou_value[class_name] = xbd_iou
            teq_iou_value[class_name] = teq_iou
        print("xbd_iou_value", xbd_iou_value, "teq_iou_value", teq_iou_value)
        self.xbd_metrics_train.reset()
        self.teq_metrics_train.reset()
        log_dict = {
            "xbd_train_mIoU": xbd_mIoU,
            "xbd_train_F1": xbd_F1,
            "xbd_train_OA": xbd_OA,
            "teq_train_mIoU": teq_mIoU,
            "teq_train_F1": teq_F1,
            "teq_train_OA": teq_OA,
            **self.get_avg_loss(self.training_step_outputs),
        }
        self.log_dict(log_dict, prog_bar=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        xbd_img, xbd_mask, teq_img, teq_mask = (
            batch["xbd_img"],
            batch["xbd_gt_semantic_seg"],
            batch["teq_img"],
            batch["teq_gt_semantic_seg"],
        )
        xbd_output = self.forward(xbd_img)
        teq_output = self.forward(teq_img)
        xbd_pre_mask = nn.Softmax(dim=1)(xbd_output["logits"])
        teq_pre_mask = nn.Softmax(dim=1)(teq_output["logits"])
        xbd_pre_mask = xbd_pre_mask.argmax(dim=1)
        teq_pre_mask = teq_pre_mask.argmax(dim=1)
        for i in range(xbd_mask.shape[0]):
            self.xbd_metrics_val.add_batch(
                xbd_mask[i].cpu().numpy(), xbd_pre_mask[i].cpu().numpy()
            )
            self.teq_metrics_val.add_batch(
                teq_mask[i].cpu().numpy(), teq_pre_mask[i].cpu().numpy()
            )

        loss_xbd_val = self.loss_xbd(xbd_output["logits"], xbd_mask)
        loss_mmd_val = self.loss_mmd(teq_output["embeddings"], xbd_output["embeddings"])
        loss_val = loss_xbd_val + loss_mmd_val

        out = {
            "loss_val": loss_val,
            "loss_xbd_val": loss_xbd_val,
            "loss_mmd_val": loss_mmd_val,
        }
        self.validation_step_outputs.append(out)

        return out

    def on_validation_epoch_end(self):
        if "vaihingen" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif "potsdam" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif "whubuilding" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif "massbuilding" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif "cropland" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            xbd_mIoU = np.nanmean(self.xbd_metrics_val.Intersection_over_Union())
            xbd_F1 = np.nanmean(self.xbd_metrics_val.F1())
            teq_mIoU = np.nanmean(self.teq_metrics_val.Intersection_over_Union())
            teq_F1 = np.nanmean(self.teq_metrics_val.F1())

        xbd_OA = np.nanmean(self.xbd_metrics_val.OA())
        xbd_iou_per_class = self.xbd_metrics_val.Intersection_over_Union()
        teq_OA = np.nanmean(self.teq_metrics_val.OA())
        teq_iou_per_class = self.teq_metrics_val.Intersection_over_Union()
        eval_value = {
            "xbd_mIoU": xbd_mIoU,
            "xbd_F1": xbd_F1,
            "xbd_OA": xbd_OA,
            "teq_mIoU": teq_mIoU,
            "teq_F1": teq_F1,
            "teq_OA": teq_OA,
        }
        print("val:", eval_value)
        xbd_iou_value = {}
        teq_iou_value = {}
        for class_name, xbd_iou, teq_iou in zip(
            self.config.classes, xbd_iou_per_class, teq_iou_per_class
        ):
            xbd_iou_value[class_name] = xbd_iou
            teq_iou_value[class_name] = teq_iou
        print("xbd_iou_value", xbd_iou_value, "teq_iou_value", teq_iou_value)
        self.xbd_metrics_val.reset()
        self.teq_metrics_val.reset()
        log_dict = {
            "xbd_val_mIoU": xbd_mIoU,
            "xbd_val_F1": xbd_F1,
            "xbd_val_OA": xbd_OA,
            "teq_val_mIoU": teq_mIoU,
            "teq_val_F1": teq_F1,
            "teq_val_OA": teq_OA,
            **self.get_avg_loss(self.validation_step_outputs),
        }
        self.log_dict(log_dict, prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):

        return self.config.train_loader

    def val_dataloader(self):

        return self.config.val_loader


# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=config.save_top_k,
        monitor=config.monitor,
        save_last=config.save_last,
        mode=config.monitor_mode,
        dirpath=config.weights_path,
        filename=config.weights_name,
    )
    logger = CSVLogger("lightning_logs", name=config.log_name)

    model = Supervision_Train(config)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(
            config.pretrained_ckpt_path, config=config
        )

    trainer = pl.Trainer(
        devices=config.gpus,
        max_epochs=config.max_epoch,
        accelerator="auto",
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=[checkpoint_callback],
        strategy="auto",
        logger=logger,
    )
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)


if __name__ == "__main__":
    main()
