import torch

def predict_with_uncertainty(
        model,x,samples=30
):

    model.train()

    preds=[]

    for _ in range(samples):

        preds.append(model(x))

    preds=torch.stack(preds)

    mean=preds.mean(0)
    std=preds.std(0)

    return mean,std