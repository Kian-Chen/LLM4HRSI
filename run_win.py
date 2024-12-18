import argparse
import os
import sys
from exp.exp_imputation import Exp_Imputation
from exp.exp_trad import Exp_Trad
import random
import numpy as np
import torch

def main():
    sys.path.append("..")
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # Basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast', help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer', help='model name, options: [Autoformer, Transformer, TimesNet]')
    
    # Data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--log_dir', type=str, default='./logs/', help='logs directory')
    parser.add_argument('--data_path', type=str, default='all1_down1.csv', help='data file')
    parser.add_argument('--log_name', type=str, default='result_imputation.txt', help='log file name')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--source_names', type=str, default='cha,par,sst', help='the name of the sources')
    parser.add_argument('--lat', type=int, default=24, help='latitude')
    parser.add_argument('--lon', type=int, default=24, help='longitude')

    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    
    # Imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    
    # Anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')
    
    # Model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--len_dff', type=int, default=256, help='dimension of L hidden state ')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    
    # Optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MAE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default= False)
    parser.add_argument('--devices', type=str, default='0,1,2', help='device ids of multile gpus')
    
    # De-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    
    # Patching
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--gpt_layers', type=int, default=6)
    parser.add_argument('--ln', type=int, default=0)
    parser.add_argument('--mlp', type=int, default=0)
    parser.add_argument('--weight', type=float, default=0)
    parser.add_argument('--percent', type=int, default=100)

    # pretrain
    parser.add_argument('--pretrain_postfix', type=str, default='checkpoint.pth', help='pretrain model path')

    # fusion layers
    parser.add_argument('--last_fusion', type=str, default='TSConv2d', help='last fusion layers setting')

    # TTTITS
    parser.add_argument('--param_sharing_strategy', type=str, default='inner_group', help='parameter sharing strategy for TTTITS')
    parser.add_argument('--d_lower', type=int, default=96, help='dimension of lower-level model')
    parser.add_argument('--ttt_style', type=str, default='TTTLinear', help='style of TTT')

    # For invert Embed
    parser.add_argument('--is_invert', type=int, default=0, help='whether to invert embedding')

    # For Expert Model
    parser.add_argument('--expert_models', type=str, default='CCSS', help='expert model, you can use W, V, A, G, M, S, H, C, D or their combination')

    # For DINEOF model
    parser.add_argument('--rank', type=int, default=5, help='rank of SVD in DINEOF')
    parser.add_argument('--tol', type=float, default=1e-8, help='tolerance in DINEOF')
    parser.add_argument('--nitemax', type=int, default=300, help='maximum number of iterations in DINEOF')
    parser.add_argument('--to_center', type=bool, default=True, help='whether to center the tensor before SVD')
    parser.add_argument('--keep_non_negative_only', type=bool, default=True, help='whether to keep non-negative values only')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    args.source_names = args.source_names.split(',')
    
    args.expert_ids = list(args.expert_models)


    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    print('Args in experiment:')
    print(args)

    if args.task_name == 'imputation':
        if args.model_id == 'DINEOF':
            Exp = Exp_Trad
        else:
            Exp = Exp_Imputation

    if args.is_training and args.model != 'DINEOF':
        for ii in range(args.itr):
            # Setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_te{}_bs{}_gl{}_dm{}_nh{}_el{}_lr{}_enci{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.train_epochs,
                args.batch_size,
                args.gpt_layers,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.learning_rate,
                args.enc_in,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)  # Set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    elif args.model != 'DINEOF':
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_te{}_bs{}_gl{}_dm{}_nh{}_el{}_lr{}_enci{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.train_epochs,
            args.batch_size,
            args.gpt_layers,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.learning_rate,
            args.enc_in,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)
        
        exp = Exp(args)  # Set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
    else:
        for ii in range(args.itr):
            setting = '{}_{}_{}_{}_Rank{}_Exp_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.rank, ii)
            exp = Exp(args)
            print('>>>>>>>start evaluating : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.evaluate(setting)


if __name__ == '__main__':
    main()
