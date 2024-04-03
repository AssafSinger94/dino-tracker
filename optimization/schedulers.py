from torch.optim.lr_scheduler import LambdaLR


def get_cnn_refiner_scheduler(optimizer, gamma=0.999, apply_every=40):
    scheduler = [lambda epoch: gamma ** (epoch // apply_every)]
    lambda_feature_refiner = lambda epochs: 1
    scheduler.append(lambda_feature_refiner)
    return LambdaLR(optimizer, lr_lambda=scheduler)
