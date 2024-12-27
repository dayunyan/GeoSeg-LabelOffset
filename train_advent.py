import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random
from tools.uda import prob_2_entropy


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
        self.automatic_optimization = False
        self.net = config.net
        self.disc_aux = config.disc_aux
        self.disc_main = config.disc_main

        self.loss_seg = config.loss_seg
        self.loss_bce = config.loss_bce
        self.alpha = config.alpha

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
        optimizer, optimizer_d_aux, optimizer_d_main = self.optimizers()
        lr_scheduler, lr_scheduler_d_aux, lr_scheduler_d_main = self.lr_schedulers()
        source_label = 1
        target_label = 0

        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LR if needed
        lr_scheduler.step()
        lr_scheduler_d_aux.step()
        lr_scheduler_d_main.step()

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in self.disc_aux.parameters():
            param.requires_grad = False
        for param in self.disc_main.parameters():
            param.requires_grad = False
        # train on source
        xbd_output = self.net(xbd_img)
        loss_seg = self.loss_seg(
            (xbd_output["logits"], xbd_output["logits_aux"]), xbd_mask
        )
        loss_seg.backward()

        # adversarial training ot fool the discriminator
        teq_output = self.net(teq_img)
        pred_trg_aux = F.interpolate(
            teq_output["logits_aux"],
            size=(teq_img.shape[2], teq_img.shape[3]),
            mode="bilinear",
        )
        d_out_aux = self.disc_aux(prob_2_entropy(F.softmax(pred_trg_aux, dim=1)))
        loss_adv_trg_aux = self.loss_bce(d_out_aux, source_label)
        pred_trg_main = F.interpolate(
            teq_output["logits"],
            size=(teq_img.shape[2], teq_img.shape[3]),
            mode="bilinear",
        )
        d_out_main = self.disc_main(prob_2_entropy(F.softmax(pred_trg_main, dim=1)))
        loss_adv_trg_main = self.loss_bce(d_out_main, source_label)
        loss_adv = loss_adv_trg_main + 0.2 * loss_adv_trg_aux
        loss_adv.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in self.disc_aux.parameters():
            param.requires_grad = True
        for param in self.disc_main.parameters():
            param.requires_grad = True
        # train with source
        pred_src_aux = xbd_output["logits_aux"].detach()
        d_out_aux = self.disc_aux(prob_2_entropy(F.softmax(pred_src_aux, dim=1)))
        loss_d_aux = self.loss_bce(d_out_aux, source_label)
        loss_d_aux = loss_d_aux / 2
        loss_d_aux.backward()
        pred_src_main = xbd_output["logits"].detach()
        d_out_main = self.disc_main(prob_2_entropy(F.softmax(pred_src_main, dim=1)))
        loss_d_main = self.loss_bce(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        # train with target
        pred_trg_aux = pred_trg_aux.detach()
        d_out_aux = self.disc_aux(prob_2_entropy(F.softmax(pred_trg_aux, dim=1)))
        loss_d_aux = self.loss_bce(d_out_aux, target_label)
        loss_d_aux = loss_d_aux / 2
        loss_d_aux.backward()
        pred_trg_main = pred_trg_main.detach()
        d_out_main = self.disc_main(prob_2_entropy(F.softmax(pred_trg_main, dim=1)))
        loss_d_main = self.loss_bce(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        optimizer_d_aux.step()
        optimizer_d_main.step()
        lr_scheduler.step()
        lr_scheduler_d_aux.step()
        lr_scheduler_d_main.step()

        xbd_pre_mask = nn.Softmax(dim=1)(xbd_output["logits"]).argmax(dim=1)
        teq_pre_mask = nn.Softmax(dim=1)(teq_output["logits"]).argmax(dim=1)
        for i in range(xbd_mask.shape[0]):
            self.xbd_metrics_train.add_batch(
                xbd_mask[i].cpu().numpy(), xbd_pre_mask[i].cpu().numpy()
            )
            self.teq_metrics_train.add_batch(
                teq_mask[i].cpu().numpy(), teq_pre_mask[i].cpu().numpy()
            )
        out = {"ls_seg": loss_seg, "ls_adv": loss_adv, "ls_d": loss_d_aux + loss_d_main}
        self.training_step_outputs.append(out)
        # return out

    def on_train_epoch_end(self):
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
            "x_t_mIoU": xbd_mIoU,
            "x_t_F1": xbd_F1,
            "x_t_OA": xbd_OA,
            "t_t_mIoU": teq_mIoU,
            "t_t_F1": teq_F1,
            "t_t_OA": teq_OA,
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
        source_label = 1
        target_label = 0

        xbd_output = self(xbd_img)
        teq_output = self(teq_img)
        xbd_pre_mask = nn.Softmax(dim=1)(xbd_output["logits"]).argmax(dim=1)
        teq_pre_mask = nn.Softmax(dim=1)(teq_output["logits"]).argmax(dim=1)
        for i in range(xbd_mask.shape[0]):
            self.xbd_metrics_val.add_batch(
                xbd_mask[i].cpu().numpy(), xbd_pre_mask[i].cpu().numpy()
            )
            self.teq_metrics_val.add_batch(
                teq_mask[i].cpu().numpy(), teq_pre_mask[i].cpu().numpy()
            )

        loss_seg_val = self.loss_seg(xbd_output["logits"], xbd_mask)
        pred_trg_main = F.interpolate(
            teq_output["logits"],
            size=(teq_img.shape[2], teq_img.shape[3]),
            mode="bilinear",
        )
        d_out_main = self.disc_main(prob_2_entropy(F.softmax(pred_trg_main, dim=1)))
        loss_adv_trg_main = self.loss_bce(d_out_main, source_label)
        loss_adv = loss_adv_trg_main
        loss_val = loss_seg_val + loss_adv

        out = {
            "loss_val": loss_val,
            "ls_seg_v": loss_seg_val,
            "ls_adv_v": loss_adv,
        }
        self.validation_step_outputs.append(out)

        return out

    def on_validation_epoch_end(self):
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
            "x_v_mIoU": xbd_mIoU,
            "x_v_F1": xbd_F1,
            "x_v_OA": xbd_OA,
            "t_v_mIoU": teq_mIoU,
            "t_v_F1": teq_F1,
            "t_v_OA": teq_OA,
            **self.get_avg_loss(self.validation_step_outputs),
        }
        self.log_dict(log_dict, prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        optimizer_d_aux = self.config.optimizer_d_aux
        lr_scheduler_d_aux = self.config.lr_scheduler_d_aux
        optimizer_d_main = self.config.optimizer_d_main
        lr_scheduler_d_main = self.config.lr_scheduler_d_main

        return [optimizer, optimizer_d_aux, optimizer_d_main], [
            lr_scheduler,
            lr_scheduler_d_aux,
            lr_scheduler_d_main,
        ]

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
