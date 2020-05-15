import os
from mxnet import nd, gluon
from colorama import Fore, Back, Style


def load_pretrained_model(net, filename, ctx):

    load_checkpoint(net, filename, ctx, allow_missing = True, ignore_extra = True)

def load_pretrained_model_only_same_shape_sup(net, filename, ctx, prefix = ''):

    pretrained = nd.load(filename); pretrained_keys = pretrained.keys()

    params = net._collect_params_with_prefix(); params_keys = params.keys()

    for p in params_keys:

        if 'A.' + p in pretrained_keys:

            try:

                print('[*] Try to load {}'.format(p))

                params[p]._load_init(pretrained['A.' + p], ctx)
                
                print('[*] Parameter {} is loaded successfully'.format(p))

            except:

                print('[!] Warning : Shape of Paramter {} {} is not consistent with the one {} in the Pretrained Model'.format(p, params[p].shape, pretrained['A.' + p].shape))

        elif p not in pretrained_keys:

            print('[!] Warning : Parameter {} is not exist in Pretrained Model'.format(p))


def load_pretrained_model_only_same_shape(net, filename, ctx, prefix = ''):

    pretrained = nd.load(filename); pretrained_keys = pretrained.keys()

    params = net._collect_params_with_prefix(); params_keys = params.keys()

    for p in params_keys:

        if p in pretrained_keys:

            try:
                params[p]._load_init(pretrained[p], ctx)
                
                print('[*] Parameter {} is loaded successfully'.format(p))

            except:

                print('[!] Warning : Shape of Paramter {} {} is not consistent with the one {} in the Pretrained Model'.format(p, params[p].shape, pretrained[p].shape))

        elif p not in pretrained_keys:

            print('[!] Warning : Parameter {} is not exist in Pretrained Model'.format(p))
            
    return 0, 0

def load_latest_checkpoint(net, directory, ctx):

    max_epcoh = -1; max_step = -1

    if not os.path.isdir(directory):
    
        print(Fore.RED + '[!] checkpoint not found in => {}'.format(directory) + Style.RESET_ALL)

        return 0, 0

    items = os.listdir(directory)

    for item in items:

        if 'params' in item:

            epoch, step = (item.split('.')[0].split('-'))

            max_epcoh = max(max_epcoh, int(epoch))

            max_step = max(max_step, int(step))

    if max_step == -1 or max_epcoh == -1:
        
        print(Fore.RED + '[!] checkpoint not found' + Style.RESET_ALL)

        return 0, 0

    checkpoint = os.path.join(directory, '{:010d}-{}.params'.format(max_epcoh, max_step))
    
    net.load_parameters(checkpoint, ctx = ctx, ignore_extra = True)

    print(Fore.LIGHTYELLOW_EX + '[*] Restore From CheckPoint => {}'.format(checkpoint) + Style.RESET_ALL)

    return max_epcoh, max_step

def load_checkpoint(net, filename, ctx, allow_missing = False, ignore_extra = False):

    net.load_parameters(filename, ctx = ctx, allow_missing = allow_missing, ignore_extra = ignore_extra)
    
    print(Fore.LIGHTYELLOW_EX + '[*] Restore From CheckPoint => {}'.format(filename) + Style.RESET_ALL)

def save_gluon_model(net, directory, epoch, step):

    if not os.path.isdir(directory):

        os.mkdir(directory)

    checkpoint = os.path.join(directory, '{:010d}-{}.params'.format(epoch, step))
    
    net.save_parameters(checkpoint)
    
    print(Fore.LIGHTYELLOW_EX + '[*] CheckPoint is save to => {}'.format(checkpoint) + Style.RESET_ALL)

def leave_only_n_lastest_checkpoint():

    pass


