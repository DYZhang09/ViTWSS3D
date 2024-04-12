import os
import sys 
import torch


if __name__ == '__main__':
    pth_file = sys.argv[1]
    assert os.path.exists(pth_file)

    state_dict = torch.load(pth_file)

    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_ckpt = {k.replace("transformer_q.", "backbone."): v for k, v in state_dict['base_model'].items()}
    remove_keys = [k for k in base_ckpt.keys() if not k.startswith('backbone')]
    [base_ckpt.pop(k) for k in remove_keys]

    # state_dict['base_model'] = base_ckpt

    out_name = os.path.splitext(pth_file)[0] + '_converted.pth'
    torch.save(base_ckpt, out_name)