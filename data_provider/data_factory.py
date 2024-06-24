from data_provider.data_loader import  Dataset_Custom, Dataset_Multisource
from torch.utils.data import DataLoader

data_dict = {
   
    'custom': Dataset_Custom,
    'multisource': Dataset_Multisource
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    
    
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent
    # TODO? shuffle is all false? why need shuffle
    # TODO? drop_last is all true? with shuffle used in loader
    # TODO? seem that bach_size is all 1
    # TODO? freq is useless
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    
    
    # TODO? seq_len = 24, label_len = 0, pred_len = 0
    data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            artificially_missing_rate = args.mask_rate,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            percent=percent,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
    )
    batch_size = args.batch_size
    print(flag, len(data_set))
    data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
    return data_set, data_loader
