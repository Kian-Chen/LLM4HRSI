from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.utils import normalization_grud as normalization
from utils.utils import renormalization_np as renormalization
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')

class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight)
        return model_optim

    def _select_criterion(self):
        criterion = nn.L1Loss()
        #criterion = nn.MSELoss()

        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        source_losses = {0: [], 1: [], 2: []}
        sources = ['cha', 'par', 'sst']
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_mask, eval_mask) in tqdm(enumerate(vali_loader)):

                batch_x = batch_x.float().to(self.device)
                batch_mask = batch_mask.to(self.device)
                eval_mask = eval_mask.to(self.device)

                batch_x = batch_x.permute(0, 2, 3, 1)
                batch_mask = batch_mask.permute(0, 2, 3, 1)
                eval_mask = eval_mask.permute(0, 2, 3, 1)
                batch_y = batch_y.permute(0, 2, 3, 1)
                
                batch_x_copy = batch_x

                if self.args.features == 'MS':
                    batch_x = batch_x.permute(0, 2, 1, 3).contiguous().view(-1, batch_x.size(1), batch_x.size(3))
                    batch_mask = batch_mask.permute(0, 2, 1, 3).contiguous().view(-1, batch_mask.size(1), batch_mask.size(3))
                    eval_mask = eval_mask.permute(0, 2, 1, 3).contiguous().view(-1, eval_mask.size(1), eval_mask.size(3))

                outputs = self.model(batch_x, batch_mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x_copy.detach().cpu()
                eval_mask = eval_mask.detach().cpu()

                losses = []
                for source_idx in range(outputs.shape[-1]):
                    outputs_source = outputs[..., source_idx]
                    batch_x_copy_source = batch_x_copy[..., source_idx]
                    eval_mask_source = eval_mask[..., source_idx]

                    loss_source = criterion(outputs_source[eval_mask_source == 1], batch_x_copy_source[eval_mask_source == 1])
                    source_losses[source_idx].append(loss_source.item())
                    losses.append(loss_source)

                loss = torch.stack(losses).mean()
                total_loss.append(loss.item())
        
        avg_total_loss = np.average(total_loss)
        avg_source_losses = {source_idx: np.average(source_losses[source_idx]) for source_idx in source_losses}

        for source_idx, avg_loss in avg_source_losses.items():
            print(f"Validation Loss for source {sources[source_idx]}: {avg_loss:.7f}")

        self.model.train()
        return avg_total_loss, avg_source_losses



    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        sources = ['cha', 'par', 'sst']

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            train_loss = []
            train_loss_sources = {0: [], 1: [], 2: []}
            iter_count = 0
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_mask, eval_mask) in tqdm(enumerate(train_loader)):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_mask = batch_mask.to(self.device)
                eval_mask = eval_mask.to(self.device)

                batch_x = batch_x.permute(0, 2, 3, 1)
                batch_mask = batch_mask.permute(0, 2, 3, 1)
                eval_mask = eval_mask.permute(0, 2, 3, 1)
                batch_y = batch_y.permute(0, 2, 3, 1)

                batch_x_copy = batch_x

                if self.args.features == 'MS':
                    batch_x = batch_x.permute(0, 2, 1, 3).contiguous().view(-1, batch_x.size(1), batch_x.size(3))
                    batch_mask = batch_mask.permute(0, 2, 1, 3).contiguous().view(-1, batch_mask.size(1), batch_mask.size(3))
                    eval_mask = eval_mask.permute(0, 2, 1, 3).contiguous().view(-1, eval_mask.size(1), eval_mask.size(3))

                outputs = self.model(batch_x, batch_mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                losses = []
                for source_idx in range(outputs.shape[-1]):
                    # 选择当前来源的数据
                    outputs_source = outputs[..., source_idx]
                    batch_x_copy_source = batch_x_copy[..., source_idx]
                    eval_mask_source = eval_mask[..., source_idx]

                    # 计算当前来源的损失
                    loss_source = criterion(outputs_source[eval_mask_source == 1], batch_x_copy_source[eval_mask_source == 1])
                    train_loss_sources[source_idx].append(loss_source.item())
                    losses.append(loss_source)
                
                valid_losses = [loss for loss in losses if not torch.isnan(loss)]
                loss = torch.stack(valid_losses).mean()
                # 将损失列表转换为张量，并求和
                #loss = torch.stack(losses).sum()

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    print("\t{0} loss: {1}, {2} loss: {3}, {4} loss: {5}".format(sources[0], losses[0].item(), sources[1], losses[1].item(), sources[2], losses[2].item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_source_losses = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_source_losses = self.vali(test_data, test_loader, criterion)

            for source_idx in range(len(sources)):
                avg_source_loss = np.average(train_loss_sources[source_idx])
                print(f"Epoch: {epoch + 1}, Source: {sources[source_idx]} | Train Loss: {avg_source_loss:.7f}  | Vali Loss: {vali_source_losses[source_idx]:.7f} | Test Loss: {test_source_losses[source_idx]:.7f}")

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = {0: [], 1: [], 2: []}
        trues = {0: [], 1: [], 2: []}
        eval_masks = {0: [], 1: [], 2: []}
        sources = ['cha', 'par', 'sst']
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_mask, eval_mask) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_mask = batch_mask.to(self.device)
                eval_mask = eval_mask.to(self.device)

                batch_x = batch_x.permute(0, 2, 3, 1)
                batch_mask = batch_mask.permute(0, 2, 3, 1)
                eval_mask = eval_mask.permute(0, 2, 3, 1)
                batch_y = batch_y.permute(0, 2, 3, 1)

                batch_x_copy = batch_x

                if self.args.features == 'MS':
                    batch_x = batch_x.permute(0, 2, 1, 3).contiguous().view(-1, batch_x.size(1), batch_x.size(3))
                    batch_mask = batch_mask.permute(0, 2, 1, 3).contiguous().view(-1, batch_mask.size(1),
                                                                                  batch_mask.size(3))
                    eval_mask = eval_mask.permute(0, 2, 1, 3).contiguous().view(-1, eval_mask.size(1),
                                                                                eval_mask.size(3))

                outputs = self.model(batch_x, batch_mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                outputs = outputs.detach().cpu().numpy()
                true = batch_x_copy.detach().cpu().numpy()
                eval_mask = eval_mask.detach().cpu().numpy()

                for source_idx in range(outputs.shape[-1]):
                    preds[source_idx].append(outputs[..., source_idx])
                    trues[source_idx].append(true[..., source_idx])
                    eval_masks[source_idx].append(eval_mask[..., source_idx])

                    if i % 20 == 0:
                        filled = true[0, :, source_idx].copy()
                        visual(true[0, :, source_idx], filled,
                               os.path.join(folder_path, f'{sources[source_idx]}_{i}.pdf'))

        for source_idx in range(len(sources)):
            preds[source_idx] = np.concatenate(preds[source_idx], 0)
            trues[source_idx] = np.concatenate(trues[source_idx], 0)
            eval_masks[source_idx] = np.concatenate(eval_masks[source_idx], 0)

        print('test shape:', {sources[source_idx]: preds[source_idx].shape for source_idx in range(len(sources))},
              {sources[source_idx]: trues[source_idx].shape for source_idx in range(len(sources))})

        raw = np.loadtxt(os.path.join(self.args.root_path, self.args.data_path), dtype=float, delimiter=",")
        data, norm_parameters = normalization(raw)

        trues1 = {source_idx: renormalization(trues[source_idx].reshape(-1, trues[source_idx].shape[-1]),
                                              norm_parameters).reshape(trues[source_idx].shape)
                  for source_idx in range(len(sources))}
        preds1 = {source_idx: renormalization(preds[source_idx].reshape(-1, preds[source_idx].shape[-1]),
                                              norm_parameters).reshape(preds[source_idx].shape)
                  for source_idx in range(len(sources))}

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_imputation.txt", 'a') as f:
            f.write(setting + "  \n")
            for source_idx in range(len(sources)):
                mae1, mse1, rmse1, mape1, mspe1 = metric(preds1[source_idx][eval_masks[source_idx] == 1],
                                                         trues1[source_idx][eval_masks[source_idx] == 1])
                mae, mse, rmse, mape, mspe = metric(preds[source_idx][eval_masks[source_idx] == 1],
                                                    trues[source_idx][eval_masks[source_idx] == 1])
                print(f'{sources[source_idx]} - mse:{mse}, mae:{mae}, rmse:{rmse}')
                f.write(
                    f'{sources[source_idx]} - rmse1:{rmse1}, mae1:{mae1}, mape1:{mape1}, rmse:{rmse}, mae:{mae}, mape:{mape}\n')

                np.save(folder_path + f'metrics_{sources[source_idx]}.npy', np.array([mae, mse, rmse, mape, mspe]))
                np.save(folder_path + f'pred_{sources[source_idx]}.npy', preds1[source_idx])
                np.save(folder_path + f'true_{sources[source_idx]}.npy', trues1[source_idx])
            f.write('\n')

        return


