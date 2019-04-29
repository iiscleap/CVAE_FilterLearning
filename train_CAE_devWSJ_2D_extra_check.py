import numpy as np
import torch
from torch.autograd import Variable

import glob
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import librosa
import time
import random
import matplotlib.pyplot as plt
import scipy.io as sio
import soundfile as sf
import scipy.fftpack as scfft
import scipy
from scipy import signal

from Generator_extra2 import VAE


OUTDIR = './'


def windows(data, window_size):
    start = 0
    while start < (len(data)):
        yield start, start + window_size
        start += int(window_size*1/10)


def extract_features(fileList, n_fft=512, hop_length=256, win_length=400, patch_length=64, bands=40):
    window_size = hop_length * (patch_length - 1) + win_length
    log_specgrams = []
    for i in range(np.alen(fileList)):
        fn = fileList[i]
        print(fn)
        sig, fs = sf.read(fn, channels=1, samplerate=16000, format='RAW', subtype='PCM_16')
        sound_clip = 0.999 * sig / max(abs(sig))
        for (start, end) in windows(sound_clip, window_size):
            if len(sound_clip[start:end]) == window_size:
                signal = sound_clip[start:end].flatten()
                no_of_frames = np.floor((np.alen(signal) - win_length) / hop_length) + 1
                sig_spec = np.zeros((int(n_fft / 2 + 1), int(no_of_frames)))
                w = scipy.signal.hamming(win_length)
                t = 0
                for kk in range(0, len(signal) - win_length, hop_length):
                    sig_spec[:, t] = np.abs(scfft.fft(w * signal[kk:kk + win_length], n_fft)[0:n_fft / 2 + 1])
                    t = t + 1

                mel_spec = librosa.feature.melspectrogram(sr=fs, S=sig_spec ** 2, n_fft=n_fft, hop_length=hop_length,
                                                          n_mels=bands, fmin=200, fmax=6500)
                mel_spec = mel_spec.transpose()
                mel_spec = np.log(mel_spec + np.finfo(np.float32).eps)
                logspec = mel_spec.flatten('F').T[np.newaxis, :]
                log_specgrams.append(logspec)
    
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), np.size(mel_spec, axis=0), np.size(mel_spec, axis=1), order='F')
    print('log_specgrams shape', np.shape(log_specgrams))
    return log_specgrams


def normalize(features, param):
    n_freq_bins = param['n_fft']/2 + 1
    mean = features.mean(axis = 1)
    mean = mean.mean(axis = 0)
    var = features.std(axis = 1)
    var = var.std(axis = 0)
    return mean, var


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    temp = mean_sq + stddev_sq - torch.log(stddev_sq + 1e-9) - 1
    temp2 = torch.mean(temp)
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq + 1e-9) - 1)


def generator_model_arch(param):
    return VAE(param=param)


def adjust_learning_rate(optimizer, lr):
    """Updates learning rate"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def batch_generator(inp_features, tar_features, param, minibatch_index):
    num_of_patches = np.size(inp_features, axis=0)
    data_inp = np.ndarray(shape=(param['batch_size'], np.size(inp_features, axis=1),
                                 np.size(inp_features, axis=2), np.size(inp_features, axis=3)), dtype=np.float32)
    data_tar = np.ndarray(shape=(param['batch_size'], np.size(inp_features, axis=1),
                                 np.size(inp_features, axis=2), np.size(inp_features, axis=3)), dtype=np.float32)
    for i in range(param['batch_size']):
        temp = minibatch_index * param['batch_size'] + i
        data_inp[i] = (inp_features[temp, :, :, :])
        data_tar[i] = (tar_features[temp, :, :, :])
    
    return torch.from_numpy(data_inp).share_memory_(), torch.from_numpy(data_tar).share_memory_()


def split_data_train_val(inp_features, train_percent=0.9):
    num_features = np.size(inp_features, axis=0)
    inp_features_shuf = np.random.permutation(inp_features)
    inp_features = inp_features_shuf[0:int(np.floor(num_features*train_percent)), :, :]
    validation_features = inp_features_shuf[int(np.floor(num_features*train_percent)):num_features-1, :, :]
    return inp_features, validation_features


def l1_penalty(var):
    return torch.abs(var).sum()


def main(param):
    reference = param['type'] + str(time.strftime("_%Y%m%d-%H%M%S"))
    inList = '/home/purvia/Purvi_new_WSJ/raw_train_clean.txt'
    with open(inList) as fin:
        infiles = fin.read().splitlines()

    tarList = '/home/purvia/Purvi_new_WSJ/raw_train_clean.txt'
    with open(tarList) as ftar:
        tarfiles = ftar.read().splitlines()

    inp_features = extract_features(infiles, n_fft=param['n_fft'], hop_length=param['hop_len'], patch_length=param['patch_length'], bands=param['bands'])

    # tar_features = extract_features(tarfiles, n_fft=param['n_fft'], hop_length=param['hop_len'], patch_length=param['patch_length'])
    inp_features, validation_features = split_data_train_val(inp_features, train_percent=0.9)

    mean, var = normalize(inp_features, param)
    statistics = {'mean': mean, 'var': var}
    inp_features = (inp_features - statistics['mean']) / (statistics['var'] + np.finfo(np.float32).eps)
    val_features_inp = (validation_features - statistics['mean']) / (statistics['var'] + np.finfo(np.float32).eps)

    inp_features = inp_features[:, :, np.newaxis, :]
    val_features_inp = val_features_inp[:, :, np.newaxis, :]
    tar_features = inp_features
    validation_features = val_features_inp
    print('inp_features, val_features', np.shape(inp_features), np.shape(validation_features))

    # For conv2d, we need to feed (B, C, H, W) - otherwise comment next 2 lines # debug
    inp_features = np.swapaxes(inp_features, 1, 2)
    tar_features = np.swapaxes(tar_features, 1, 2)
    val_features_inp = np.swapaxes(val_features_inp, 1, 2)
    validation_features = np.swapaxes(validation_features, 1, 2)

    num_minibatches_per_epoch = np.size(inp_features, axis=0) / param['batch_size']
    print ('Num minibatches per epoch: {}'.format(num_minibatches_per_epoch))

    G_net = generator_model_arch(param=param)

    L2_criterion = torch.nn.MSELoss()

    if param['cuda']:
        G_net.cuda()
        L2_criterion.cuda()

    G_optimizer = torch.optim.Adam(G_net.parameters(), lr=param['lr_G'])

    train_err_G_L2 = np.zeros(shape=(param['num_epochs'],))
    training_error_G_filtConstraint_r = np.zeros(shape=(param['num_epochs'],))
    training_error_G_filtConstraint_s = np.zeros(shape=(param['num_epochs'],))
    training_error_G_L1penalty_enc = np.zeros(shape=(param['num_epochs'],))
    training_error_G_LatentLoss = np.zeros(shape=(param['num_epochs'],))
    train_err_G = np.zeros(shape=(param['num_epochs'],))

    # Validation errors
    training_error_G_L2_val = np.zeros(shape=(param['num_epochs'],))
    training_error_G_L1penalty_enc_val = np.zeros(shape=(param['num_epochs'],))
    training_error_G_LatentLoss_val = np.zeros(shape=(param['num_epochs'],))
    training_error_G_val = np.zeros(shape=(param['num_epochs'],))

    epoch = 0
    current_lr_G = param['lr_G']
    current_lr_D = param['lr_D']

    while epoch < param['num_epochs']:
        print('epoch no.:', epoch)
        start_time = time.time()
        minibatch_process_time = 0
        G_net.train()

        minibatches_seen = 0
        while minibatches_seen < num_minibatches_per_epoch:
            inputs, targets = batch_generator(inp_features, tar_features, param, minibatches_seen)

            mbatch_time = time.time()

            # Send to GPU
            if param['cuda']:
                inputs = inputs.float().cuda()
                targets = targets.float().cuda()

            # train generator
            G_net.zero_grad()
            fake_enc, fake = G_net(Variable(inputs))
	    r1 = G_net.encoder.r1.data.cpu().numpy()
            r1 = np.flipud(r1)
	    r2 = G_net.encoder.r2.data.cpu().numpy()
            r2 = np.flipud(r2)

	    s1 = G_net.encoder.s1.data.cpu().numpy()
            s1 = np.flipud(s1)
	    s2 = G_net.encoder.s2.data.cpu().numpy()
            s2 = np.flipud(s2)

            err_filtConstraint_r = np.linalg.norm(np.convolve(r1, r2), 2)
	    err_filtConstraint_s = np.linalg.norm(np.convolve(s1, s2), 2)

            errL2 = L2_criterion(fake, Variable(targets))
	    errLatentLoss = latent_loss(G_net.z_mean, G_net.z_sigma)
            errG = 1.0 * errL2 + 0.1 * (err_filtConstraint_r + err_filtConstraint_s) + 0.1 * l1_penalty(fake_enc) + 0.01 * errLatentLoss
            errG.backward(retain_graph=True)
            G_optimizer.step()

            train_err_G_L2[epoch] += errL2.data[0]
            train_err_G[epoch] += errG.data[0]
            training_error_G_filtConstraint_r[epoch] += err_filtConstraint_r
	    training_error_G_filtConstraint_s[epoch] += err_filtConstraint_s
            training_error_G_L1penalty_enc[epoch] += l1_penalty(fake_enc)
	    training_error_G_LatentLoss[epoch] += errLatentLoss.data[0]

            minibatches_seen += 1
	
	# VALIDATION
        print("Computing validation loss:\t\t ")
        if param['validate']:
            G_net.eval()  # Set model to evaluate mode
            minibatches_seen_val = 0
            num_minibatches_per_epoch_val = np.size(val_features_inp, axis=0) / param['batch_size']
            while minibatches_seen_val < num_minibatches_per_epoch_val:
                val_inputs, val_targets = batch_generator(val_features_inp, validation_features, param, minibatches_seen_val)
                # print(val_inputs.shape, val_targets.shape)
		if param['cuda']:
                    val_inputs = val_inputs.float().cuda()
                    val_targets = val_targets.float().cuda()

                fake_enc_val, fake_val = G_net(Variable(val_inputs))

                errL2_val = L2_criterion(fake_val, Variable(val_targets))
                errLatentLoss_val = latent_loss(G_net.z_mean, G_net.z_sigma)
                errG_val = 1.0 * errL2_val + 0.1 * (err_filtConstraint_r + err_filtConstraint_s) + 0.1 * l1_penalty(fake_enc_val) + 0.01 * errLatentLoss_val
                training_error_G_L2_val[epoch] += errL2_val.data[0]
                training_error_G_L1penalty_enc_val[epoch] += l1_penalty(fake_enc_val)
                training_error_G_LatentLoss_val[epoch] += errLatentLoss_val.data[0]
                training_error_G_val[epoch] += errG_val.data[0]
                minibatches_seen_val += 1

            training_error_G_L2_val[epoch] /= num_minibatches_per_epoch_val
            training_error_G_L1penalty_enc_val[epoch] /= num_minibatches_per_epoch_val
            training_error_G_LatentLoss_val[epoch] /= num_minibatches_per_epoch_val
            training_error_G_val[epoch] /= num_minibatches_per_epoch_val
            print('Done validation\t\t')


        train_err_G_L2[epoch] /= num_minibatches_per_epoch
        training_error_G_filtConstraint_r[epoch] /= num_minibatches_per_epoch
	training_error_G_filtConstraint_s[epoch] /= num_minibatches_per_epoch
        training_error_G_L1penalty_enc[epoch] /= num_minibatches_per_epoch
	training_error_G_LatentLoss[epoch] /= num_minibatches_per_epoch
        train_err_G[epoch] /= num_minibatches_per_epoch
        time_for_1_epoch = time.time() - start_time

        # Print Training Error every epoch
        print("Epoch {} of {} took {:.3f}s (training time {:.2f}s, Data prep {:.2f}s".format(
            epoch + 1, param['num_epochs'], time_for_1_epoch, minibatch_process_time,
            time_for_1_epoch - minibatch_process_time))
        print("train_loss:\t\tL2:{:.4f}\t\terr_filtCons_r:{:.8f}\t\terr_filtCons_s:{:.8f}\t\terr_L1penalty:{:.4f}\t\tG_LatentLoss:{:.4f}\t\tG:{:.4f}".format(train_err_G_L2[epoch],
                                        training_error_G_filtConstraint_r[epoch], training_error_G_filtConstraint_s[epoch], training_error_G_L1penalty_enc[epoch], 
					training_error_G_LatentLoss[epoch], train_err_G[epoch]))

        
	if param['validate']:
            print("Validation loss:\t\tL2:{:.4f} \t\terr_L1penalty_enc:{:.4f}\t\tLatenLoss: {:.4f}\t\tG: {:.4f}".format(training_error_G_L2_val[epoch],
                                training_error_G_L1penalty_enc_val[epoch], training_error_G_LatentLoss_val[epoch], training_error_G_val[epoch]))
	
	sys.stdout.flush()

        print('Saving current model ...')
        if not os.path.exists(OUTDIR + reference):
            os.mkdir(OUTDIR + reference)
        model_path_G = OUTDIR + reference + '/model_G_' + str(epoch) + '.pth'
        if (epoch + 1) % 2 == 0:
            torch.save(G_net.state_dict(), model_path_G)

	 # Update learning late as per learning rate schedule
        if (epoch + 1) % param['LearningRateEpoch'] == 0 and epoch != 0:
            if current_lr_G >= 1e-5:
                current_lr_G *= 0.1
                adjust_learning_rate(G_optimizer, (current_lr_G))
                print ('===================Updated G Learning Rate to {}====================='.format(current_lr_G))

        sio.savemat(OUTDIR + reference + '/training_error_G_L2.mat', mdict={'data': train_err_G_L2})
        sio.savemat(OUTDIR + reference + '/training_error_G_filtConstraint_rate.mat', mdict={'data': training_error_G_filtConstraint_r})
	sio.savemat(OUTDIR + reference + '/training_error_G_filtConstraint_scale.mat', mdict={'data': training_error_G_filtConstraint_s})
        sio.savemat(OUTDIR + reference + '/training_error_G_L1penalty_enc.mat', mdict={'data': training_error_G_L1penalty_enc})
	sio.savemat(OUTDIR + reference + '/training_error_G_LatentLoss.mat', mdict={'data': training_error_G_LatentLoss})
        sio.savemat(OUTDIR + reference + '/train_err_G.mat', mdict={'data': train_err_G})

	if param['validate']:
            sio.savemat(OUTDIR + reference + '/training_error_G_L2_val.mat', mdict={'data': training_error_G_L2_val})
            sio.savemat(OUTDIR + reference + '/training_error_G_L1penalty_enc_val.mat', mdict={'data': training_error_G_L1penalty_enc_val})
            sio.savemat(OUTDIR + reference + '/training_error_G_val.mat', mdict={'data': training_error_G_val})
            sio.savemat(OUTDIR + reference + '/training_error_G_LatentLoss_val.mat', mdict={'data': training_error_G_LatentLoss_val})

	
        # Update Epoch
        epoch += 1
    return model_path_G

if __name__ == '__main__':
    Param = {
        'type': 'vae_generator',
        'bands': 40,
        'n_fft': 512,
        'patch_length': 150,
        'hop_len': 160,
        'win_length': 400,
        'G_type': 'vae',
        'n_channels': 1,
	'num_nodes_fnn': [6000, 6000, 6000, 6000, 6000], # For 2D if mel bands=40, (then 40 x 150) with 5x5 kernel (padding of (2,2) for VAE
	'num_nodes_mean_var': 5000,
        'input_nc': 1,
        'output_nc': 1,
        'input_nc_disc': 2,
        'ngf': 2,  # of gen. filters
        'filt_h': 5,  # kernel height
        'filt_w': 5,  # kernel width
        'norm_layer': 'nn.BatchNorm2d',
        'use_dropout': 'False',
        'gpu_ids': 1,
        'lr_G': 0.001,
        'lr_D': 0.0001,
	'LearningRateEpoch': 10,
        'num_epochs': 60,
        'cuda': True,
        'batch_size': 10,
	'validate': True,
    }

model_path_G = main(param=Param)
