import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss
from transformers import BertTokenizer
import torch.nn.functional as F

logger = logging.getLogger('MMSA')

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class DMD():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()
        self.crossentropy = nn.CrossEntropyLoss()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def do_train(self, model, dataloader, return_epoch_results=False):

        # 0: DMD model, 1: Homo GD, 2: Hetero GD
        params = list(model[0].parameters())

        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = []
        net_dmd = model[0]

        net.append(net_dmd)

        model = net

        while True:
            epochs += 1
            y_pred, y_true = [], []
            for mod in model:
                mod.train()

            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train_lb']) as td:
                for batch_data in td:

                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data[1].to(self.args.device)
                    text = self.tokenizer(batch_data[2], return_tensors='pt', padding=True, truncation=True)
                    text = {key: value.to(self.args.device) for key, value in text.items()}
                    labels = batch_data[3].to(self.args.device)
                    labels = labels.view(-1, 1)


                    output = model[0](text, vision)

                    # decouple loss
                    # reconstruction loss
                    loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                    loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                    loss_recon = loss_recon_l + loss_recon_v

                    # cycle consistency loss between s_x and s_x_r
                    loss_sl_slr = self.MSE(output['s_l'], output['s_l_r'])
                    loss_sv_slv = self.MSE(output['s_v'], output['s_v_r'])
                    loss_s_sr = loss_sl_slr + loss_sv_slv

                    # ort loss
                    cosine_similarity_s_c_l = self.cosine(output['s_l'], output['c_l'],
                                                          torch.tensor([-1]).cuda()).mean(0)
                    cosine_similarity_s_c_v = self.cosine(output['s_v'], output['c_v'],
                                                          torch.tensor([-1]).cuda()).mean(0)
                    loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v

                    # margin loss
                    c_l, c_v = output['c_l_sim'], output['c_v_sim']
                    ids, feats = [], []
                    for i in range(labels.size(0)):
                        feats.append(c_l[i].view(1, -1))
                        feats.append(c_v[i].view(1, -1))
                        # feats.append(c_a[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        # ids.append(labels[i].view(1, -1))
                    feats = torch.cat(feats, dim=0)
                    ids = torch.cat(ids, dim=0)
                    loss_sim = self.sim_loss(ids, feats)

                    decouple_loss = loss_s_sr + loss_recon + (loss_sim+loss_ort) * 0.1 

                    # classification loss
                    loss_t_c = self.crossentropy(output['pre_t'],labels.squeeze())
                    loss_v_c = self.crossentropy(output['pre_v'],labels.squeeze())
                    loss_m_c = self.crossentropy(output['pre_m'],labels.squeeze())
                    classification_loss = loss_t_c + loss_v_c + loss_m_c


                    loss = decouple_loss + classification_loss

                    loss.backward()


                    if self.args.grad_clip != -1.0:
                        params = list(model[0].parameters())
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)

                    train_loss += decouple_loss.item()

                    y_pred.append(output['pre_m'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                    # break
                if not left_epochs:
                    # update
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train_lb'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model[0], dataloader['eval'], mode="VAL")
            test_results = self.do_test(model[0], dataloader['eval'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            # save each epoch model
            torch.save(model[0].state_dict(), './pt/' + str(epochs) + '.pth')
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                model_save_path = './pt/dmd.pth'
                torch.save(model[0].state_dict(), model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):

        model.eval()
        y_pred, y_true = [], []

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data[1].to(self.args.device)
                    text = self.tokenizer(batch_data[2], return_tensors='pt', padding=True, truncation=True)
                    text = {key: value.to(self.args.device) for key, value in text.items()}
                    labels = batch_data[3].to(self.args.device)
                    labels = labels.view(-1, 1)
                    output = model(text, vision)
                    loss = self.criterion(output['pre_m'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['pre_m'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results
    