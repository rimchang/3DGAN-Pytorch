import torch
from torch import optim
from torch import  nn
from collections import OrderedDict
from utils import make_hyparam_string, save_new_pickle, read_pickle, SavePloat_Voxels, generateZ
import os


from utils import ShapeNetDataset, var_or_cuda
from model import _G, _D
from lr_sh import  MultiStepLR

def train(args):

    hyparam_list = [("model", args.model_name),
                    ("cube", args.cube_len),
                    ("bs", args.batch_size),
                    ("g_lr", args.g_lr),
                    ("d_lr", args.d_lr),
                    ("z", args.z_dis),
                    ("bias", args.bias),
                    ("sl", args.soft_label),]

    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyparam_list))
    log_param = make_hyparam_string(hyparam_dict)
    print(log_param)

    # for using tensorboard
    if args.use_tensorboard:
        import tensorflow as tf

        summary_writer = tf.summary.FileWriter(args.output_dir + args.log_dir + log_param)

        def inject_summary(summary_writer, tag, value, step):
                summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
                summary_writer.add_summary(summary, global_step=step)

        inject_summary = inject_summary


    # datset define
    dsets_path = args.input_dir + args.data_dir + "train/"
    print(dsets_path)
    dsets = ShapeNetDataset(dsets_path, args)
    dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # model define
    D = _D(args)
    G = _G(args)

    D_solver = optim.Adam(D.parameters(), lr=args.d_lr, betas=args.beta)
    G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)

    if args.lrsh:
        D_scheduler = MultiStepLR(D_solver, milestones=[500, 1000])

    if torch.cuda.is_available():
        print("using cuda")
        D.cuda()
        G.cuda()

    criterion = nn.BCELoss()

    pickle_path = "." + args.pickle_dir + log_param
    read_pickle(pickle_path, G, G_solver, D, D_solver)

    for epoch in range(args.n_epochs):
        for i, X in enumerate(dset_loaders):

            X = var_or_cuda(X)

            if X.size()[0] != int(args.batch_size):
                #print("batch_size != {} drop last incompatible batch".format(int(args.batch_size)))
                continue

            Z = generateZ(args)
            real_labels = var_or_cuda(torch.ones(args.batch_size))
            fake_labels = var_or_cuda(torch.zeros(args.batch_size))

            if args.soft_label:
                real_labels = var_or_cuda(torch.Tensor(args.batch_size).uniform_(0.7, 1.2))
                fake_labels = var_or_cuda(torch.Tensor(args.batch_size).uniform_(0, 0.3))

            # ============= Train the discriminator =============#
            d_real = D(X)
            d_real_loss = criterion(d_real, real_labels)


            fake = G(Z)
            d_fake = D(fake)
            d_fake_loss = criterion(d_fake, fake_labels)

            d_loss = d_real_loss + d_fake_loss


            d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
            d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu),0))

            if d_total_acu <= args.d_thresh:
                D.zero_grad()
                d_loss.backward()
                D_solver.step()

            # =============== Train the generator ===============#

            Z = generateZ(args)

            fake = G(Z)
            d_fake = D(fake)
            g_loss = criterion(d_fake, real_labels)

            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            G_solver.step()

        # =============== logging each iteration ===============#
        iteration = str(G_solver.state_dict()['state'][G_solver.state_dict()['param_groups'][0]['params'][0]]['step'])
        if args.use_tensorboard:
            log_save_path = args.output_dir + args.log_dir + log_param
            if not os.path.exists(log_save_path):
                os.makedirs(log_save_path)

            info = {
                'loss/loss_D_R': d_real_loss.data[0],
                'loss/loss_D_F': d_fake_loss.data[0],
                'loss/loss_D': d_loss.data[0],
                'loss/loss_G': g_loss.data[0],
                'loss/acc_D' : d_total_acu.data[0]
            }

            for tag, value in info.items():
                inject_summary(summary_writer, tag, value, iteration)

            summary_writer.flush()

        # =============== each epoch save model or save image ===============#
        print('Iter-{}; , D_loss : {:.4}, G_loss : {:.4}, D_acu : {:.4}, D_lr : {:.4}'.format(iteration, d_loss.data[0], g_loss.data[0], d_total_acu.data[0], D_solver.state_dict()['param_groups'][0]["lr"]))

        if (epoch + 1) % args.image_save_step == 0:

            samples = fake.cpu().data[:8].squeeze().numpy()

            image_path = args.output_dir + args.image_dir + log_param
            if not os.path.exists(image_path):
                os.makedirs(image_path)

            SavePloat_Voxels(samples, image_path, iteration)

        if (epoch + 1) % args.pickle_step == 0:
            pickle_save_path = args.output_dir + args.pickle_dir + log_param
            save_new_pickle(pickle_save_path, iteration, G, G_solver, D, D_solver)

        if args.lrsh:

            try:

                D_scheduler.step()


            except Exception as e:

                print("fail lr scheduling", e)
