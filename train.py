import argparse
from Brainmets.utils import *
from Brainmets.dataset import *
from Brainmets.augmentations import Transformer
from Brainmets.losses import *
from Brainmets.trainer import *
from Brainmets.evaluation import *

if __name__== '__main__':
    
    """
    data_name: Param("dataset to run on", str) ['Baseline']
    loss: Param("loss function for training", str) = ['Diceloss', 'BCE', 'Focal', 'WBCE']
    suffix : Param("anything", str) = 'anything you want to add on the name of the saved model',
    gpu: Param("gpu to train on", str) ['1','2',....],
    debug: Param("debug mode", bool) = [True, False]
    
    example:
        run by command line: python train.py Baseline BCE example_1 0 True
        the result name will be Baseline-BCE-example_1 in the results folder
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="num of epochs to train", type=int)
    parser.add_argument("--name", help="data mode")
    parser.add_argument("--suffix", help="anything to add on the end of the name of the saved model")
    parser.add_argument("--gpu", help="gpu to train on")
    parser.add_argument("--max_lr", help="max learning rate", type=float)
    parser.add_argument("--loss", help="loss function to use")
    parser.add_argument("--pos_weight", help="pos_weight for WBCE", type=float)
    parser.add_argument("--debug", help="debug mode or not")
    args = parser.parse_args()
    
    loss = args.loss
    pos_weight = args.pos_weight
    suffix = args.suffix
    gpu = args.gpu
    debug = args.debug
    debug_size = 20
    init_lr = 0.001
    max_lr = args.max_lr
    final_div_factor = 100
    epochs = args.epochs
    print_per_instance = True
    use_one_cycle = True
    name = '-'.join([args.name, loss, suffix, str(max_lr)])

    device = torch.device('cuda:' + gpu)

    data_path = Path('../data')
    data = 'manuscript_1_datasets_first_tx_allmets'
    df = pd.read_csv(data_path/f'{data}.csv')

    train_df = df[df['split'] == 'train'].sample(frac=1)
    valid_df = df[df['split'] == 'valid'].sample(frac=1)
    test_df = df[df['split'] == 'test'].sample(frac=1)

    train_img_files = list(train_df['img_files'])
    train_mask_files = list(train_df['mask_files'])
    valid_img_files = list(valid_df['img_files'])
    valid_mask_files = list(valid_df['mask_files'])
    test_img_files = list(test_df['img_files'])
    test_mask_files = list(test_df['mask_files'])

    img_files = sorted(train_img_files + valid_img_files + test_img_files)
    mask_files = sorted(train_mask_files + valid_mask_files + test_mask_files)
    img_names = ['_'.join(file.split('/')[-1].split('_')[0:2])
                 for file in img_files]
    mask_names = ['_'.join(file.split('/')[-1].split('_')[0:2])
                  for file in mask_files]
    assert img_names == mask_names

    # train_transformer = Transformer(axes=['d', 'h', 'w'], max_zoom_rate=1.5, angle=15)
    train_transformer = None
    valid_transformer = None
    test_transformer = None

    if debug == 'True':
        train_dataset = MetDataSet(
            train_df.iloc[:debug_size], train_transformer)
        valid_dataset = MetDataSet(valid_df.iloc[:debug_size])
        test_dataset = MetDataSet(test_df.iloc[:debug_size])
    else:
        train_dataset = MetDataSet(train_df, train_transformer)
        valid_dataset = MetDataSet(valid_df)
        test_dataset = MetDataSet(test_df)

    print('train data size: ', len(train_dataset))
    print('valid data size: ', len(valid_dataset))
    print('test data size: ', len(test_dataset))

    if loss == 'Diceloss':
        loss_func = DiceLoss().to(device)
    elif loss == 'BCE':
        loss_config = {'name': 'BCEWithLogitsLoss'}
        config = {'loss': loss_config}
        loss_func = get_loss_criterion(config).to(device)
    elif loss == 'BCE_my':
        loss_func = FocalLossLogits().to(device)
    elif loss == 'Focal':
        loss_func = FocalLossLogits(pos_weight=1, gamma=0.5).to(device)
    elif loss == 'WBCE_my':
        loss_func = FocalLossLogits(pos_weight=pos_weight).to(device)
    elif loss == 'WBCE':
        pos_weight = torch.tensor(pos_weight)
        loss_config = {
            'name': 'WeightedBCEWithLogitsLoss',
            'pos_weight': pos_weight}
        config = {'loss': loss_config}
        loss_func = get_loss_criterion(config).to(device)

    trainer = Trainer(
        name,
        'ResidualUNet3D',
        train_dataset,
        valid_dataset,
        test_dataset,
        1,
        init_lr,
        max_lr,
        loss_func,
        device)

    trainer.fit(epochs, print_per_instance, use_one_cycle)

    trainer = Trainer.load_best_checkpoint(name)

    test_score = trainer.predict()
