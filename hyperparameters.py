class Hyperparameters(object):
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

hp = Hyperparameters(
    n_epochs=10,    
    dataset_train_mode="train",
    dataset_test_mode="test", 
    batch_size=4,        
    lr=.0002,
    decay_start_epoch=5,
    b1=.5,
    b2=0.999,
    n_cpu=8,
    img_size=256,
    channels=3,
    n_critic=5,
    sample_interval=200,
    num_residual_blocks=10,
    lambda_cyc=10.0,
    lambda_id=5.0)