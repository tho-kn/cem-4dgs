#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os
import pprint

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true" if not value else "store_false")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true" if not value else "store_false")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = True
        self.model = "virtual"
        self.loader = "neural3dvideo" 
        self.interp_type = "cube" 
        self.rot_interp_type = "slerp"
        self.connectivity_threshold = 0.5
        self.lazy_loader = True
        self.llffhold = 8
        self.time_interval = 5
        self.time_pad = 3
        self.var_pad = 3
        self.time_pad_type = 0 # 0: none, 1: reflect 2: repeat
        self.kernel_size = 0.1
        self.start_duration = 5
        self.duration = -1
        self.sample_every = 1
        self.progressive_step = 1
        self.start_timestamp = 0
        self.end_timestamp = -1
        self.near = 0.2
        self.far = 300.0
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.dynamic_position_lr_init = 0.00016
        self.dynamic_position_lr_final = 0.000016
        self.dynamic_position_lr_delay_mult = 0.01
        self.dynamic_position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.00001
        self.disp_lr = 0.0001
        self.feature_motion_lr = 0.0025
        self.rotation_motion_lr = 0.001
        self.opacity_center_lr = 0.001
        self.opacity_var_lr = 0.0005
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.l1_accum = True
        self.densification_interval = 200
        self.densify_from_iter = 500
        self.extract_from_iter = 500
        self.densify_until_iter = 15_000
        self.progressive_growing_steps = 300
        self.error_base_prune_steps = 20000
        self.ssim_prune_every = 5
        self.l1_prune_every = 5
        self.make_dynamic_interval = 200
        self.extraction_interval = 3000
        self.extract_percentile = 0.98
        self.child_extract_percentile = 0.98
        self.prune_invisible_interval = 6000
        self.densify_grad_threshold = 0.0002
        self.densify_dgrad_threshold = 0.0001
        self.disp_sim_threshold = 0.0
        self.s_max_ssim = 0.6
        self.s_l1_thres = 0.08
        self.d_max_ssim = 0.6
        self.d_l1_thres = 0.08
        self.static_reg = 0.0001
        self.motion_reg = 0.0001
        self.rot_reg = 0.00
        self.coord_reg = 0.00
        self.cec_gaussian_blur_size = 0
        self.random_background = True
        self.size_threshold = 1e10
        self.dynamic_size_threshold = 1e10
        self.extract_motion_thres = 1000.0
        self.extract_min_motion_thres = 1e-6
        self.extract_motion_thres_child = 1000.0
        self.extract_min_motion_thres_child = 1e-6
        self.cec_every = -1
        self.cec_start_iter = 1e8
        self.cec_until_iter = 6000
        self.cec_depth_kernel = 1
        self.cec_margin_depth = 1
        self.cec_error_abs_thres = 0.02
        self.cec_error_rel_thres = 0.05
        self.cec_opacity_threshold = 0.9
        self.cec_dynamic_leniency = -1.0
        self.cec_nearest_threshold = 0.5 # Previously tested with 0.01
        super().__init__(parser, "Optimization Parameters")
        
def find_explicit_args(args):
    args_specified = set()
    skip_next = False
    for i, arg in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if arg.startswith('--'):
            if '=' in arg:
                arg_name = arg.split('=')[0].strip('-').replace('-', '_')
                args_specified.add(arg_name)
            else:
                arg_name = arg.strip('-').replace('-', '_')
                args_specified.add(arg_name)
                skip_next = True  # Skip the next item, which is the value
    return args_specified

def get_combined_args(parser : ArgumentParser, cfgfilepath : str = None):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)
    
    # Identify explicitly given arguments
    args_specified = find_explicit_args(args_cmdline)
    try:
        if cfgfilepath is None:
            cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass

    args_cfgfile = eval(cfgfile_string)
    if isinstance(args_cfgfile, dict):
        args_cfgfile = Namespace(**args_cfgfile)
    else:
        with open(os.path.join(args_cmdline.model_path, "cfg_args_dict"), 'w') as cfg_log_f:
            cfg_log_f.write(pprint.pformat(vars(args_cfgfile)))

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if k not in merged_dict:
            merged_dict[k] = v
        if k in args_specified and v is not None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
