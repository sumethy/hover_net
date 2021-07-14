"""run_infer.py

Usage:
  run_infer.py [options] [--help] <command> [<args>...]
  run_infer.py --version
  run_infer.py (-h | --help)

Options:
  -h --help                   Show this string.
  --version                   Show version.

  --gpu=<id>                  GPU list. [default: 0]
  --nr_types=<n>              Number of nuclei types to predict. [default: 0]
  --type_info_path=<path>     Path to a json define mapping between type id, type name, 
                              and expected overlaid color. [default: '']

  --model_path=<path>         Path to saved checkpoint.
  --model_mode=<mode>         Original HoVer-Net or the reduced version used PanNuke and MoNuSAC, 
                              'original' or 'fast'. [default: fast]
  --nr_inference_workers=<n>  Number of workers during inference. [default: 8]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 16]
  --batch_size=<n>            Batch size per 1 GPU. [default: 32]

Two command mode are `tile` and `wsi` to enter corresponding inference mode
    tile  run the inference on tile
    wsi   run the inference on wsi

Use `run_infer.py <command> --help` to show their options and usage.
"""

tile_cli = """
Arguments for processing tiles.

usage:
    tile (--input_dir=<path>) (--output_dir=<path>) \
         [--draw_dot] [--save_qupath] [--save_raw_map] [--mem_usage=<n>] [--mrxs_path=<path>]
    
options:
   --input_dir=<path>     Path to input data directory. Assumes the files are not nested within directory.
   --output_dir=<path>    Path to output directory..

   --mem_usage=<n>        Declare how much memory (physical + swap) should be used for caching. 
                          By default it will load as many tiles as possible till reaching the 
                          declared limit. [default: 0.2]
   --draw_dot             To draw nuclei centroid on overlay. [default: False]
   --save_qupath          To optionally output QuPath v0.2.3 compatible format. [default: False]
   --save_raw_map         To save raw prediction or not. [default: False]
   --mrxs_path=<path>            Path to mrxs file
"""

wsi_cli = """
Arguments for processing wsi

usage:
    wsi (--input_dir=<path>) (--output_dir=<path>) [--proc_mag=<n>]\
        [--cache_path=<path>] [--input_mask_dir=<path>] \
        [--ambiguous_size=<n>] [--chunk_shape=<n>] [--tile_shape=<n>] \
        [--save_thumb] [--save_mask]
    
options:
    --input_dir=<path>      Path to input data directory. Assumes the files are not nested within directory.
    --output_dir=<path>     Path to output directory.
    --cache_path=<path>     Path for cache. Should be placed on SSD with at least 100GB. [default: cache]
    --mask_dir=<path>       Path to directory containing tissue masks. 
                            Should have the same name as corresponding WSIs. [default: '']

    --proc_mag=<n>          Magnification level (objective power) used for WSI processing. [default: 40]
    --ambiguous_size=<int>  Define ambiguous region along tiling grid to perform re-post processing. [default: 128]
    --chunk_shape=<n>       Shape of chunk for processing. [default: 10000]
    --tile_shape=<n>        Shape of tiles for processing. [default: 2048]
    --save_thumb            To save thumb. [default: False]
    --save_mask             To save mask. [default: False]
"""

import torch
from torch.utils.data import DataLoader, Dataset

import os
import re
import sys
import numpy as np
import pickle

import logging
import os
import copy
from misc.utils import log_info
from docopt import docopt
import openslide
from tqdm import tqdm

from infer.base import InferManager
import matplotlib.pyplot as plt
import json
from importlib import import_module
from run_utils.utils import convert_pytorch_checkpoint

import cv2
from skimage import measure
from PIL import Image
import uuid
resnet101_input_size = 256
um_per_px = 0.1538
size_threshold = 5.13
num_img = 296

class MyDataSet(Dataset):
    def __init__(self, mrxs_path):
        super(MyDataSet).__init__()
        self.img = openslide.OpenSlide(mrxs_path)
        self.resnet101_input_size = 256
        self.tiles = self.getTileList()

    def getTileList(self):
        tiles = []
        bound_x = int(self.img.properties["openslide.bounds-x"])
        bound_y = int(self.img.properties["openslide.bounds-y"])
        bound_w = int(self.img.properties["openslide.bounds-width"])
        bound_h = int(self.img.properties["openslide.bounds-height"])

        num_tiles_row = bound_h // self.resnet101_input_size
        num_tiles_col = bound_w // self.resnet101_input_size
        for row in range(0, num_tiles_row): 
            for col in range(0, num_tiles_col):
                coord = (col*self.resnet101_input_size + bound_x, row*self.resnet101_input_size + bound_y)
                tiles.append(coord)

        return tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        tile = self.img.read_region(
            (self.tiles[index][0],self.tiles[index][1]), 
            0, 
            (self.resnet101_input_size, self.resnet101_input_size)) # PIL image
        array = np.array(tile) # ndarray
        array = array[...,:3]
        # array = np.swapaxes(array,0,2)

        return {
            "input_tensor": torch.tensor(array, dtype=torch.uint8),
            "x" : self.tiles[index][0],
            "y": self.tiles[index][1]
        }

class MyInferManager(object):

    def __init__(self, mrxs_path, **kwargs):
        self.mrxs_path = mrxs_path
        self.dataset = MyDataSet(self.mrxs_path)
        self.dataloader = DataLoader(self.dataset,batch_size=32,num_workers=32)

        self.filename_prefix =  self.mrxs_path.split('/')[-1].split('.')[0]
        self.filename_prefix = re.sub("-", "_", self.filename_prefix)
        self.filename_prefix += "_"

        self.run_step = None
        for variable, value in kwargs.items():
            self.__setattr__(variable, value)
        self.load_model()
        self.nr_types = self.method["model_args"]["nr_types"]
        # create type info name and colour

        # default
        self.type_info_dict = {
            None: ["no label", [0, 0, 0]],
        }

        if self.nr_types is not None and self.type_info_path is not None:
            self.type_info_dict = json.load(open(self.type_info_path, "r"))
            self.type_info_dict = {
                int(k): (v[0], tuple(v[1])) for k, v in self.type_info_dict.items()
            }
            # availability check
            for k in range(self.nr_types):
                if k not in self.type_info_dict:
                    assert False, "Not detect type_id=%d defined in json." % k

        if self.nr_types is not None and self.type_info_path is None:
            cmap = plt.get_cmap("hot")
            colour_list = np.arange(self.nr_types, dtype=np.int32)
            colour_list = (cmap(colour_list)[..., :3] * 255).astype(np.uint8)
            # should be compatible out of the box wrt qupath
            self.type_info_dict = {
                k: (str(k), tuple(v)) for k, v in enumerate(colour_list)
            }

        # self.net = self.load_model()

    def load_model(self):
        """Create the model, load the checkpoint and define
        associated run steps to process each data batch.
        
        """
        model_desc = import_module("models.hovernet.net_desc")
        model_creator = getattr(model_desc, "create_model")

        net = model_creator(**self.method["model_args"])
        saved_state_dict = torch.load(self.method["model_path"])["desc"]
        saved_state_dict = convert_pytorch_checkpoint(saved_state_dict)

        net.load_state_dict(saved_state_dict, strict=True)
        net = torch.nn.DataParallel(net)
        net = net.to("cuda")

        module_lib = import_module("models.hovernet.run_desc")
        run_step = getattr(module_lib, "infer_step")
        self.run_step = lambda input_batch: run_step(input_batch, net)

        module_lib = import_module("models.hovernet.post_proc")
        self.post_proc_func = getattr(module_lib, "process")
        # return net

    def getROI(self, original, mask, coord_x, coord_y):
        # TODO call object detection model on original
        global num_img
        already_saved_orignal = False
        # mask is 2D ndarray: 0 (background), 1,2,3,4. we want only 1 (red) cell
        blobs = mask == 1 # array of boolean
        blobs_labels = measure.label(blobs, background=0)
        num_components = np.max(blobs_labels) # number of cells detected for this image

        mask4eachComponent = np.zeros([resnet101_input_size,resnet101_input_size,num_components+1],dtype=np.uint8) # channel 1 for object 1, channel 2 for object 2, etc. channel 0 unused
        for obj in range(1,num_components+1):
            mask4eachComponent[:,:,obj] = (mask == obj)*255

        cb_detected = []
        
        def isCellCB(contours):
            if len(contours) == 0:
                return False
            area = cv2.contourArea(contours[0])
            equi_diameter = np.sqrt(4*area/np.pi)*um_per_px
            # check ratio of major/miner axis (how round it is)
            rect = cv2.boundingRect(contours[0])
            aspectRatio = rect[2] / rect[3]

            if equi_diameter > size_threshold and (0.7 <= aspectRatio and aspectRatio <= 1.3):
                return True
            else:
                return False

        counter = 1
        for obj in range(1,num_components+1):
            ret,thresh = cv2.threshold(mask4eachComponent[:,:,obj],127,255,0)
            contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            if isCellCB(contours):
                if not already_saved_orignal: # save original only have at least 1 CB was detected (and save only once!)
                    # original.save('/centroblast/tmp/wsiScanResult/image' + str(num_img).zfill(4) + '.png')
                    original.save('/centroblast/tmp/wsiScanResult/' + self.filename_prefix + str(coord_x).zfill(7) + "_" + str(coord_y).zfill(7) + '.png')
                    num_img += 1
                    already_saved_orignal = True

                # find the centroid 
                M = cv2.moments(contours[0])
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # cb_detected.append((obj,cX,cY,mask4eachComponent[:,:,obj]))
                bbox = [cX-45,cY-45,cX+45,cY+45]
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = min(255, bbox[2])
                bbox[3] = min(255, bbox[3])
                roi = original.crop(bbox)
                # roi.save('/tmp/centroblast/' + str(uuid.uuid4()) + '.png')
                # roi.save('/centroblast/tmp/wsiScanResult/image' + str(num_img-1).zfill(4) + '_ROI' +str(counter).zfill(4)+'_'+str(bbox[0]).zfill(3)+'_'+str(bbox[1]).zfill(3)+'_'+str(bbox[2]).zfill(3)+'_'+str(bbox[3]).zfill(3)+'.png')
                roi.save('/centroblast/tmp/wsiScanResult/' + self.filename_prefix + str(coord_x).zfill(7) + "_" + str(coord_y).zfill(7) + '_ROI' +str(counter).zfill(4)+'_'+str(bbox[0]).zfill(3)+'_'+str(bbox[1]).zfill(3)+'_'+str(bbox[2]).zfill(3)+'_'+str(bbox[3]).zfill(3)+'.png')
                counter += 1
                
                # TODO call classification model on roi

        # return cb_detected


    def post_process_cb(self, batch, output_batch):
        # for each image in batch
            # slice the first channel (the segmentation mask)
            # resize to 256,256
            # detect cb (consider red blobs only for speed)
            # if detected save image (for now, actually have to send to the classification model)
        
        for b in range(output_batch.shape[0]):
            seg_mask = output_batch[b,:,:,0] # the first channel is the segmentation mask
            seg_mask = cv2.resize(seg_mask, (256,256)) # https://stackoverflow.com/questions/55428929/
            original_image = batch["input_tensor"][b].detach().numpy()
            original_image = Image.fromarray(original_image)
            coord_x = batch["x"][b].detach().numpy()
            coord_y = batch["y"][b].detach().numpy()
            # original_image.save('/tmp/centroblast/' + str(uuid.uuid4()) + '.png')
            self.getROI(original_image, seg_mask, coord_x, coord_y)

        # print(type(batch))
        # print(batch.shape)
        # print(type(output_batch))
        # print(output_batch.shape)

    def process_tiles(self, run_args):
        print("number of samples",len(self.dataset))
        for i, batch in tqdm(enumerate(self.dataloader)):
            # print(batch)
            output_batch = self.run_step(batch["input_tensor"]) # first channel of output is probably the segmentation mask (channel last format)
            # print(output_batch.shape)
            self.post_process_cb(batch, output_batch)
            # break


if __name__ == '__main__':
    sub_cli_dict = {'tile' : tile_cli, 'wsi' : wsi_cli}
    args = docopt(__doc__, help=False, options_first=True, 
                    version='HoVer-Net Pytorch Inference v1.0')
    sub_cmd = args.pop('<command>')
    sub_cmd_args = args.pop('<args>')

    # ! TODO: where to save logging
    logging.basicConfig(
        level=logging.INFO,
        format='|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s',datefmt='%Y-%m-%d|%H:%M:%S',
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )

    if args['--help'] and sub_cmd is not None:
        if sub_cmd in sub_cli_dict: 
            print(sub_cli_dict[sub_cmd])
        else:
            print(__doc__)
        exit()
    if args['--help'] or sub_cmd is None:
        print(__doc__)
        exit()

    sub_args = docopt(sub_cli_dict[sub_cmd], argv=sub_cmd_args, help=True)
    
    args.pop('--version')
    gpu_list = args.pop('--gpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    nr_gpus = torch.cuda.device_count()
    log_info('Detect #GPUS: %d' % nr_gpus)

    args = {k.replace('--', '') : v for k, v in args.items()}
    sub_args = {k.replace('--', '') : v for k, v in sub_args.items()}
    if args['model_path'] == None:
        raise Exception('A model path must be supplied as an argument with --model_path.')

    nr_types = int(args['nr_types']) if int(args['nr_types']) > 0 else None
    method_args = {
        'method' : {
            'model_args' : {
                'nr_types'   : nr_types,
                'mode'       : args['model_mode'],
            },
            'model_path' : args['model_path'],
        },
        'type_info_path'  : None if args['type_info_path'] == '' \
                            else args['type_info_path'],
    }

    # ***
    run_args = {
        'batch_size' : int(args['batch_size']) * nr_gpus,

        'nr_inference_workers' : int(args['nr_inference_workers']),
        'nr_post_proc_workers' : int(args['nr_post_proc_workers']),
    }

    if args['model_mode'] == 'fast':
        run_args['patch_input_shape'] = 256
        run_args['patch_output_shape'] = 164
    else:
        run_args['patch_input_shape'] = 270
        run_args['patch_output_shape'] = 80

    if sub_cmd == 'tile':
        run_args.update({
            'input_dir'      : sub_args['input_dir'],
            'output_dir'     : sub_args['output_dir'],

            'mem_usage'   : float(sub_args['mem_usage']),
            'draw_dot'    : sub_args['draw_dot'],
            'save_qupath' : sub_args['save_qupath'],
            'save_raw_map': sub_args['save_raw_map'],
            'mrxs_path'   : sub_args['mrxs_path']
        })

    if sub_cmd == 'wsi':
        run_args.update({
            'input_dir'      : sub_args['input_dir'],
            'output_dir'     : sub_args['output_dir'],
            'input_mask_dir' : sub_args['input_mask_dir'],
            'cache_path'     : sub_args['cache_path'],

            'proc_mag'       : int(sub_args['proc_mag']),
            'ambiguous_size' : int(sub_args['ambiguous_size']),
            'chunk_shape'    : int(sub_args['chunk_shape']),
            'tile_shape'     : int(sub_args['tile_shape']),
            'save_thumb'     : sub_args['save_thumb'],
            'save_mask'      : sub_args['save_mask'],
        })
    # ***
    
    if sub_cmd == 'tile':
        print(run_args)
        infer = MyInferManager(run_args['mrxs_path'], **method_args)
        infer.process_tiles(run_args)
    else:
        from infer.wsi import InferManager
        infer = InferManager(**method_args)
        infer.process_wsi_list(run_args)


"""
example how to run this file:
python my_infer.py --gpu=0,1 --nr_types=5 --type_info_path=type_info.json --model_path=checkpoint/hovernet_fast_monusac_type_tf2pytorch.tar --batch_size=64 tile --input_dir=/centroblast/tmp/image --mrxs_path=/centroblast/tmp/data/CB-001-1.mrxs --output_dir=/centroblast/tmp/output
"""