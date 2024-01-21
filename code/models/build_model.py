from .unet_model import UNet



def build_model(args):
    # choose different Neural network model for different args


    if args.model == "Unet":
        model = UNet(n_channels=args.input_channel, n_classes=args.n_classes, bilinear=True, bias=True)

    else:
        raise

    return model
