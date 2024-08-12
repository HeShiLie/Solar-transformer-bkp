import torch
import numpy as np
import pickle
import json
import argparse
from types import SimpleNamespace

import os
import random
import time

from Model.encoder import Embeddings,PretrainModel
from Model.decoder import LinearDecoder

from Solarclip_test_tq import calculate_loss_pretrain
from Data.Solardataloader_tq import enhance_funciton
from Data.utils_tq import transfer_date_to_id
import Data.Solardataloader_subset_tq as Solardataloader_subset_tq



def main():
    encoder = Embeddings(
        in_channels=1,
        input_resolution=1024,
        patch_size=64,
        width=768,
        layers=12,
        heads=12,
        output_dim=768,
        token_type='class embedding'
    )
    decoder = LinearDecoder(
        input_dim=768,
        output_dim=1024
    )
    pretrainModel = PretrainModel(encoder, decoder)

    checkpoint_path = '/mnt/nas/home/huxing/202407/ctf/SolarCLIP/checkpoints/pretrain/'
    logger_checkpoint_path = checkpoint_path + 'logger/'
    model_checkpoint_path = checkpoint_path + 'model/'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    start_date = transfer_date_to_id(2010, 5, 1)
    end_date = transfer_date_to_id(2020, 6, 30)
    train_loader = Solardataloader_subset_tq.get_loader_by_time(time_interval=[
                                                             start_date, end_date], modal_list=["magnet","0094"], load_imgs = False, enhance_list=[1024,0.5,90], batch_size=400, shuffle=True, num_workers=16)
    start_date = transfer_date_to_id(2020, 6, 30)
    end_date = transfer_date_to_id(2024, 6, 30)
    val_loader = Solardataloader_subset_tq.get_loader_by_time(time_interval=[
                                                           start_date, end_date], modal_list=["magnet","0094"], load_imgs = False, enhance_list=[1024,0,0], batch_size=400, shuffle=True, num_workers=16)
    print(f"DataLoader time: {(time.time()-start_time)/60:.2f} min")

    optimizer = torch.optim.SGD(
        pretrainModel.parameters(), lr=0.1, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100)

    epochs = 100
    test_epoch = epochs//10
    save_epoch = epochs//2

    modal = 'magnet'

    logger_train_loss = []
    logger_lr = []
    logger_val_loss = []

    start_time = time.time()
    print('Start training')
    for epoch in range(epochs):
        PretrainModel.train()
        epoch_time = time.time()
        for i, data in enumerate(train_loader):
            data = data.to(device)
            if modal == 'magnet':
                image = data[:,0,:,:,:]
            elif modal == '0094':
                image = data[:,1,:,:,:]
            image = enhance_funciton(image, "None", 1)
            iteration_txt = f"Iteration {i} | Data time: {
                (time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            loss = calculate_loss_pretrain(modal, encoder, decoder, image)
            iteration_txt += f" Forward time: {
                (time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration_txt += f" Backward time: {
                (time.time()-epoch_time)/60:.2f} min |"
            print(iteration_txt)
            epoch_time = time.time()

            logger_train_loss.append(loss.item())
            logger_lr.append(scheduler.get_last_lr()[0])
        scheduler.step()

        print(f'Epoch {epoch+1:>6}/{epochs:<6} | Train loss {logger_train_loss[-1]:<10.8f} | LR {logger_lr[-1]:<10.8f} | Cost {(time.time()-start_time)/60:.2f} min')

        if (epoch+1) % test_epoch == 0:
            with torch.no_grad():
                PretrainModel.eval()
                loss_results = []
                
                for i, data in enumerate(val_loader):
                    data = data.to(device)
                    if modal == 'magnet':
                        image = data[:,0,:,:,:]
                    elif modal == '0094':
                        image = data[:,1,:,:,:]
                    image = enhance_funciton(image, "None", 1)
                    loss = calculate_loss_pretrain(modal, encoder, decoder, image)
                    
                    logger_train_loss.append(loss.item())
                    logger_lr.append(scheduler.get_last_lr()[0])

                logger_val_loss.append(np.mean(loss_results))
                
                result_txt = f'Epoch {epoch+1:>6}/{epochs:<6} | '

                result_txt += f'Train loss {logger_train_loss[-1]:<10.8f} | '
                result_txt += f'Val loss {logger_val_loss[-1]:<10.8f} | '
                result_txt += f'Save time {(time.time()-start_time)/60:.2f} min'
                print(result_txt)

                with open(f'{logger_checkpoint_path}logger_train_loss.pkl', 'wb') as f:
                    pickle.dump(logger_train_loss, f)
                with open(f'{logger_checkpoint_path}logger_lr.pkl', 'wb') as f:
                    pickle.dump(logger_lr, f)
                with open(f'{logger_checkpoint_path}logger_val_loss.pkl', 'wb') as f:
                    pickle.dump(logger_val_loss, f)

        if (epoch+1) % save_epoch == 0:
            torch.save({'model': PretrainModel.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'epoch': epoch},
                       f'{model_checkpoint_path}epoch_{epoch+1}.pt')
            print(f'Model saved {(epoch+1)/epochs:.2%}, cost {(time.time()-start_time)/60:.2f} min')

if __name__ == '__main__':
    main()  