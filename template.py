def set_template(args):
    if 'jpeg' in args.template:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.lr_decay = 100

    if 'EDSR_paper' in args.template:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

    if 'MDSR' in args.template:
        args.model = 'MDSR'
        args.patch_size = 48
        args.epochs = 650

    if 'DDBPN' in args.template:
        args.model = 'DDBPN'
        args.patch_size = 128
        args.scale = '4'
        args.data_test = 'Set5'
        args.batch_size = 20
        args.epochs = 1000
        args.lr_decay = 500
        args.gamma = 0.1
        args.weight_decay = 1e-4
        args.loss = '1*MSE'

    if 'GAN' in args.template:
        args.epochs = 200
        args.lr = 5e-5
        args.lr_decay = 150
