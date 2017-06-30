import torch
from torch import optim
from torch import  nn
from collections import OrderedDict
from utils import make_hyparam_string, save_new_pickle, read_pickle, SavePloat_Voxels, generateZ
import os


from utils import ShapeNetDataset, var_or_cuda
from model import _G, _D

def test(args):

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


    # model define
    D = _D(args)
    G = _G(args)

    D_solver = optim.Adam(D.parameters(), lr=args.d_lr, betas=args.beta)
    G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)

    if torch.cuda.is_available():
        print("using cuda")
        D.cuda()
        G.cuda()

    criterion = nn.BCELoss()

    pickle_path = "." + args.pickle_dir + log_param
    read_pickle(pickle_path, G, G_solver, D, D_solver)

    for iteration in range(args.test_iter):

        Z = generateZ(args)

        fake = G(Z).eval()


        samples = fake.cpu().data[:8].squeeze().numpy()

        image_path = args.output_dir + '/test/' + log_param
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        SavePloat_Voxels(samples, image_path, iteration)


        pickle_save_path = args.output_dir + args.pickle_dir + log_param
        save_new_pickle(pickle_save_path, iteration, G, G_solver, D, D_solver)

