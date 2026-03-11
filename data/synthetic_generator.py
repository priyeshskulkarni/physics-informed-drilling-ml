import numpy as np
import pandas as pd

def generate_dataset(samples=50000):

    temperature = np.random.uniform(80,180,samples)
    pressure = np.random.uniform(3000,15000,samples)
    flow_rate = np.random.uniform(20,60,samples)
    density = np.random.uniform(1.0,2.2,samples)
    shear_rate = np.random.uniform(50,500,samples)

    tau_y = 5
    K = 0.8
    n = 0.6

    viscosity = tau_y + K * (shear_rate ** n)

    noise = np.random.normal(0,0.5,samples)

    viscosity = viscosity + noise

    df = pd.DataFrame({
        "temperature":temperature,
        "pressure":pressure,
        "flow_rate":flow_rate,
        "density":density,
        "shear_rate":shear_rate,
        "viscosity":viscosity
    })

    df.to_csv("data/raw/drilling_data.csv",index=False)

    print("Dataset generated:",df.shape)

if __name__=="__main__":
    generate_dataset()