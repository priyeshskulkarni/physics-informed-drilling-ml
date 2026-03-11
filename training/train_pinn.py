import torch
import pandas as pd
from models.model_registry import get_model
from physics.physics_constraints import physics_loss
from training.loss_functions import total_loss

df=pd.read_csv("data/raw/drilling_data.csv")

X=df[[
    "temperature",
    "pressure",
    "flow_rate",
    "density",
    "shear_rate"
]].values

y=df["viscosity"].values

X=torch.tensor(X,dtype=torch.float32)
y=torch.tensor(y,dtype=torch.float32).view(-1,1)

model=get_model()

optimizer=torch.optim.Adam(
    model.parameters(),lr=0.001
)

epochs=200

for epoch in range(epochs):

    pred=model(X)

    data_loss=torch.mean((pred-y)**2)

    phys_loss=physics_loss(
        pred,X[:,4].view(-1,1)
    )

    loss=total_loss(data_loss,phys_loss)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch%20==0:

        print(
            epoch,
            loss.item()
        )

torch.save(model.state_dict(),"model.pt")