import itertools
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torchvision.utils import make_grid
from hyperparameters import Hyperparameters
from dataset import ImageDataset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from model import ReplayBuffer, GeneratorResNet,Discriminator, LambdaLR
from torchvision.utils import save_image
from utils import initialize_conv_weights_normal,plot_output
from tqdm import tqdm
import pickle
import argparse


def save_img_samples(epoch):
    """Saves a generated sample from the test set"""
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    imgs = next(iter(val_dataloader))
    Gen_AB.eval()
    Gen_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = Gen_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = Gen_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=16, normalize=True)
    real_B = make_grid(real_B, nrow=16, normalize=True)
    fake_A = make_grid(fake_A, nrow=16, normalize=True)
    fake_B = make_grid(fake_B, nrow=16, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)    
    path =  "outputs-{}.png".format(epoch)

    save_image(image_grid, path, normalize=False)    
    return path 


def train(name,Gen_BA,Gen_AB,Disc_A,Disc_B,train_dataloader,n_epochs,criterion_identity,
          criterion_cycle,lambda_cyc,criterion_GAN,optimizer_G,fake_A_buffer,fake_B_buffer,
          optimizer_Disc_A,optimizer_Disc_B,Tensor,lambda_id):
    
    # TRAINING
    disc_loss = 0
    gen_loss = 0
    id_loss = 0
    disc_loss_total,gen_loss_total, id_loss_total = [],[],[]
    for epoch in range(n_epochs):
        for batch in tqdm(train_dataloader):

            # Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(
                Tensor(np.ones((real_A.size(0), *Disc_A.module.output_shape))),
                requires_grad=False,
            )
            fake = Variable(
                Tensor(np.zeros((real_A.size(0), *Disc_A.module.output_shape))),
                requires_grad=False,
            )

            #########################
            #  Train Generators
            #########################

            Gen_AB.module.train() # Gen_AB(real_A) will take real_A and produce fake_B
            Gen_BA.module.train() # Gen_BA(real_B) will take real_B and produce fake_A

            optimizer_G.zero_grad()

            # Identity loss
            # First pass real_A images to the Genearator, that will generate A-domains images
            loss_id_A = criterion_identity(Gen_BA(real_A), real_A)
            

            # Then pass real_B images to the Genearator, that will generate B-domains images
            loss_id_B = criterion_identity(Gen_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2
            id_loss += loss_identity.item()

            # GAN losses for GAN_AB
            fake_B = Gen_AB(real_A)
            loss_GAN_AB = criterion_GAN(Disc_B(fake_B), valid)

            # GAN losses for GAN_BA
            fake_A = Gen_BA(real_B)
            loss_GAN_BA = criterion_GAN(Disc_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle Consistency losses
            reconstructed_A = Gen_BA(fake_B)

            loss_cycle_A = criterion_cycle(reconstructed_A, real_A)

            reconstructed_B = Gen_AB(fake_A)

            loss_cycle_B = criterion_cycle(reconstructed_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity
            gen_loss+=loss_G.item()

            loss_G.backward()

            optimizer_G.step()

            #########################
            #  Train Discriminator A
            #########################

            optimizer_Disc_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(Disc_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(Disc_A(fake_A_.detach()), fake)

            loss_Disc_A = (loss_real + loss_fake) / 2
#             disc_loss_A += loss_Disc_A.item()
            loss_Disc_A.backward()

            optimizer_Disc_A.step()

            #########################
            #  Train Discriminator B
            #########################

            optimizer_Disc_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(Disc_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(Disc_B(fake_B_.detach()), fake)
            loss_Disc_B = (loss_real + loss_fake) / 2

            loss_Disc_B.backward()

            optimizer_Disc_B.step()

            loss_D = (loss_Disc_A + loss_Disc_B) / 2
            disc_loss+= loss_D.item()

        gen_loss = gen_loss/len(train_dataloader)
        disc_loss = disc_loss/len(train_dataloader)
        id_loss = id_loss/len(train_dataloader)
        gen_loss_total.append(gen_loss)
        disc_loss_total.append(disc_loss)
        id_loss_total.append(id_loss)
        plot_output(save_img_samples(epoch), 30, 40)
        
        path = "./checkpoint"
        if os.path.exists(path) is not True:
            os.mkdir(path)
        path = path + "/"+name+".pt"
        torch.save({
                    'epoch': epoch,
                    'Gen_AB': Gen_AB.state_dict(),
                    'Gen_BA': Gen_BA.state_dict(),
                    'Disc_A': Disc_A.state_dict(),
                    'Disc_B': Disc_B.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'optimizer_Disc_A': optimizer_Disc_A.state_dict(),
                    'optimizer_Disc_B': optimizer_Disc_B.state_dict()}, path)
        print(
                "\r[Epoch %d/%d] [Disc loss: %f] [Gen loss: %f] [Identity loss: %f]"
                % (
                    epoch+1,
                    n_epochs,
                    disc_loss,
                    gen_loss,
                    id_loss,
                )
            )
    losses = {"gen_loss": gen_loss_total,"disc_loss": disc_loss_total,"id_loss": id_loss_total}
    with open('outputs/losses.pickle', 'wb') as handle:
        pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

##training ends
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, default="CycleGan_VanGogh_Checkpoint", help="Name of model to be saved.")
    parser.add_argument("--data_dir_A", default="dataset/vangogh2photo/trainA", type=str, 
                        help="Directory of Van Gogh pictures/Image data A.")
    parser.add_argument("--data_dir_B", default="dataset/vangogh2photo/trainB", type=str, 
                        help="Directory of Van Gogh pictures/Image data B.")
    parser.add_argument("--val_data_dir_A", default="dataset/vangogh2photo/testA", type=str,
                        help="validation data directory for image data A.")
    parser.add_argument("--val_data_dir_B", default="dataset/vangogh2photo/testB", type=str,
                        help="validation data directory for image data B.")
    parser.add_argument("--epochs", default=10,type=int, help="Number of epochs. Best to use 200 as discussed in paper.")
    parser.add_argument("--lr", default=.0002, type=int, help= "Learning rate")
    parser.add_argument("--decay_start_epoch", default= 5, type=int, help="Epoch number where decay starts.")
    parser.add_argument("--num_residual_blocks",default=9,type=int, help="Number of residual blocks in CycleGAN generator.")
    parser.add_argument("--img_size", default=256, type=int, help="Dimension of the image. Training image must be nxn.")
    parser.add_argument("--batch_size", default= 4, type= int, help= "Batch size for training.")
    args = parser.parse_args()

    hp = Hyperparameters(name = args.name, n_epochs = args.epochs, batch_size = args.batch_size, lr = args.lr, decay_start_epoch = args.decay_start_epoch,
                         b1 = 0.5,b2 = 0.999, img_size = args.img_size, channels = 3, num_residual_blocks = args.num_residual_blocks,
                         lambda_cyc = 10.0, lambda_id = 5.0)

    print("Hyperparameters: \n")
    print(hp)

    train_transforms_ = [
        transforms.Resize((286, 286)),
        transforms.RandomRotation(degrees=(0,180)),
        transforms.RandomCrop(size=(hp.img_size,hp.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    val_transforms_ = [
        transforms.Resize((hp.img_size, hp.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    train_dataloader = DataLoader(
        ImageDataset(root=[args.data_dir_A,args.data_dir_B], transforms_=train_transforms_),
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=2)
    
    val_dataloader = DataLoader(
        ImageDataset(root= [args.val_data_dir_A,args.val_data_dir_B], transforms_=val_transforms_),
        batch_size=8,
        shuffle=True,
        num_workers=2)
    def to_img(x):
        x = x.view(x.size(0)*2, hp.channels, hp.img_size, hp.img_size)
        return x

    cuda = True if torch.cuda.is_available() else False
    print("Using CUDA" if cuda else "Not using CUDA")
    if cuda is False:
        exit("CUDA is necessary to train the model.")
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
 
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    
    input_shape = (hp.channels, hp.img_size, hp.img_size)
    
    # Initialize generator and discriminator
    Gen_AB = GeneratorResNet(input_shape, hp.num_residual_blocks)
    Gen_BA = GeneratorResNet(input_shape, hp.num_residual_blocks)
    Disc_A = Discriminator(input_shape)
    Disc_B = Discriminator(input_shape)
    
    if cuda:
        Gen_AB = nn.DataParallel(Gen_AB)
        Gen_AB = Gen_AB.cuda()
        Gen_BA = nn.DataParallel(Gen_BA)
        Gen_BA = Gen_BA.cuda()
        Disc_A = nn.DataParallel(Disc_A)
        Disc_A = Disc_A.cuda()
        Disc_B = nn.DataParallel(Disc_B)
        Disc_B = Disc_B.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()
        
    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()
    optimizer_G = torch.optim.Adam(
    itertools.chain(Gen_AB.parameters(), Gen_BA.parameters()), lr=hp.lr, betas=(hp.b1, hp.b2))
    optimizer_Disc_A = torch.optim.Adam(Disc_A.parameters(), lr=hp.lr, betas=(hp.b1, hp.b2))
    optimizer_Disc_B = torch.optim.Adam(Disc_B.parameters(), lr=hp.lr, betas=(hp.b1, hp.b2))
    
    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(hp.n_epochs, 0, hp.decay_start_epoch).step)
    lr_scheduler_Disc_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_Disc_A, lr_lambda=LambdaLR(hp.n_epochs, 0, hp.decay_start_epoch).step)
    lr_scheduler_Disc_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_Disc_B, lr_lambda=LambdaLR(hp.n_epochs, 0, hp.decay_start_epoch).step)
    
    checkpoint = torch.load("checkpoint\CycleGan_VanGogh_Checkpoint.pt") if os.path.exists("checkpoint\CycleGan_VanGogh_Checkpoint.pt") else None
    if checkpoint is not None:
        print("Loading checkpoint...")
        Gen_AB.load_state_dict(checkpoint['Gen_AB'])
        Gen_BA.load_state_dict(checkpoint['Gen_BA'])
        Disc_A.load_state_dict(checkpoint['Disc_A'])
        Disc_B.load_state_dict(checkpoint['Disc_A'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_Disc_A.load_state_dict(checkpoint['optimizer_Disc_A'])
        optimizer_Disc_B.load_state_dict(checkpoint['optimizer_Disc_B'])
        print("Successfully loaded checkpoint.")
        
    else:
        # Initialize weights
        Gen_AB.apply(initialize_conv_weights_normal)
        Gen_BA.apply(initialize_conv_weights_normal)
        Disc_A.apply(initialize_conv_weights_normal)
        Disc_B.apply(initialize_conv_weights_normal)

    train(name = hp.name, Gen_BA = Gen_BA,Gen_AB = Gen_AB,Disc_A = Disc_A,Disc_B = Disc_B,train_dataloader = train_dataloader,
          n_epochs = hp.n_epochs,criterion_identity = criterion_identity,criterion_cycle = criterion_cycle, lambda_cyc = hp.lambda_cyc,
          criterion_GAN = criterion_GAN,optimizer_G = optimizer_G,fake_A_buffer = fake_A_buffer,fake_B_buffer = fake_B_buffer,
          optimizer_Disc_A = optimizer_Disc_A,optimizer_Disc_B = optimizer_Disc_B,Tensor = Tensor, lambda_id = hp.lambda_id)