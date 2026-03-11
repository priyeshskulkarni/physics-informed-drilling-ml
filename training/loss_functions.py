def total_loss(data_loss,physics_loss,alpha=0.3):

    return data_loss + alpha*physics_loss