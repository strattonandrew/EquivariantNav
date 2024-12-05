#import torch.nn.functional as F

from .srnn_model import *

#from .equi_utils import *

#from torch import nn
#import torch
#from .pytorch_utils import *

from escnn import gspaces
from escnn import nn as enn

import escnn.nn.init as init

WIDTH = 2.
N_RINGS = 3
DEEP = False
POOL = True

NARROW = 8
MID = 16
WIDE = 32

def construct_fc_edge_index(seq_length=1, nenv=16, human_num=20):
    num_edges_per_graph = (human_num + 1) * nenv
    prod = torch.cartesian_prod(torch.arange(human_num+1), torch.arange(human_num+1))
    prod_list = [prod]
    for i in range(1, nenv):
        #print(i)
        prod_add = prod + i * (human_num+1)
        #print(prod_add)
        prod_list.append(prod_add)
    prod = torch.cat(prod_list, dim=0)
    prod = prod.T

    seq_list = []
    for i in range(seq_length):
        prod_add = prod + i * (nenv * (human_num+1))
        seq_list.append(prod_add)
    prod_seq = torch.cat(seq_list, dim=1)
    #prod_seq = torch.tile(prod, (1, seq_length))
    #print("PROD: ", prod.shape, prod_seq.shape)
    print("PROD SEQ FC ", prod_seq.shape)
    print(prod_seq)
    return prod_seq

def construct_disconnected_edge_index(seq_length=1, nenv=16, human_num=20):
    num_edges_per_graph = human_num + 1
    num_edges = num_edges_per_graph * nenv * seq_length
    edges = torch.arange(num_edges)
    edges = torch.tile(edges, (2, 1))
    #print("EDGES ", edges.shape, edges)
    print("DISCONNECTED EDGES ", edges.shape)
    print(edges)
    return edges

'''
class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers, batch_first, bias,
                 width=1, height=1,
                 kernel_size=1, stride=1,
                 aux_weights=True):
        super().__init__()

        self._hidden_size = hidden_size
        self._input_size = input_size
        self._width = width
        self._height = height
        self._stride = stride
        self._kernel_size = kernel_size
        self._aux_weights = aux_weights

        # Initialize weights for Hadamard Products
        if self._aux_weights:
            self.W_ci = nn.Parameter(torch.zeros(1, hidden_size, height, width))
            self.W_co = nn.Parameter(torch.zeros(1, hidden_size, height, width))
            self.W_cf = nn.Parameter(torch.zeros(1, hidden_size, height, width))

    def init_group(self, group_helper):
        num_rotations = group_helper.num_rotations
        grp_act = group_helper.grp_act
        scaler = group_helper.scaler
        self.in_type = enn.FieldType(grp_act,
                                     (self._input_size + self._hidden_size)
                                     // num_rotations // scaler
                                     * group_helper.reg_repr)

        divisor = num_rotations // 4

        self.out_type = enn.FieldType(grp_act, self._hidden_size // divisor // scaler
                                      * group_helper.reg_repr)

        self.c_type = enn.FieldType(grp_act, self._hidden_size
                                    // num_rotations // scaler
                                    * group_helper.reg_repr)

        self._conv = enn.R2Conv(self.in_type,
                                self.out_type,
                                stride=self._stride,
                                padding=self._kernel_size//2,
                                kernel_size=1)

    def forward(self, inputs, states=None):
        seq_len = inputs.size(0)
        batch_size = inputs.size(1)
        num_channel = inputs.size(2)
        height = self._height
        width = self._width

        # seq_len (S), batch_size (B), num_channel (C), height (H), width (W)
        inputs = inputs.reshape(seq_len, batch_size,
                                num_channel, height, width)

        if states is None:
            c = torch.zeros((batch_size, self._hidden_size, height,
                             width), dtype=torch.float).to(ptu.device)
            h = torch.zeros((batch_size, self._hidden_size, height,
                             width), dtype=torch.float).to(ptu.device)
        else:
            h, c = states

            if isinstance(h, enn.GeometricTensor):
                h = h.tensor
                c = c.tensor

        outputs = []
        for t in range(seq_len):
            x = inputs[t, ...]
            cat_x = torch.cat([x, h], dim=1)
            cat_x = self.in_type(cat_x)
            conv_x = self._conv(cat_x)

            # chunk across channel dimension
            i, f, g, o = torch.chunk(conv_x.tensor, chunks=4, dim=1)

            if self._aux_weights:
                i = torch.sigmoid(i + self.W_ci*c)
                f = torch.sigmoid(f + self.W_cf*c)
            else:
                i = torch.sigmoid(i)
                f = torch.sigmoid(f)

            c = f*c + i*torch.tanh(g)

            if self._aux_weights:
                o = torch.sigmoid(o + self.W_co*c)
            else:
                o = torch.sigmoid(o)
            h = o*torch.tanh(c)

            outputs.append(h)

        outputs = torch.stack(outputs)

        outputs = outputs.reshape(seq_len*batch_size, -1, height, width)

        return self.c_type(outputs), (self.c_type(h), self.c_type(c))

class actorRNN(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        action_embedding_size,
        obs_embedding_size,
        rnn_hidden_size,
        policy_layers,
        rnn_num_layers,
        group_helper=None,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.group_helper = group_helper

        num_rotations = self.group_helper.num_rotations
        scaler = self.group_helper.scaler
        grp_act = self.group_helper.grp_act

        # (1) trivial for g, (2) irrep for xy, (2) trivial for z, r
        in_type = enn.FieldType(grp_act, self.group_helper.irr_repr)
        
        out_type = enn.FieldType(grp_act, action_embedding_size //
                                    num_rotations//scaler*self.group_helper.reg_repr)
        self.action_embedder = utl.EquiFeatureExtractor(in_type, out_type)
'''
class GroupHelper():
    def __init__(self, num_rotations, flip_symmetry=True):
        self.num_rotations = num_rotations

        if flip_symmetry:
            self.grp_act = gspaces.flipRot2dOnR2(num_rotations)
            self.scaler = 2
        else:
            self.grp_act = gspaces.rot2dOnR2(num_rotations)
            self.scaler = 1

        self.reg_repr = [self.grp_act.regular_repr]
        self.triv_repr = [self.grp_act.trivial_repr]
        if flip_symmetry:
            self.irr_repr = [self.grp_act.irrep(1, 1)]
        else:
            self.irr_repr = [self.grp_act.irrep(1)]

class EquiLinear(nn.Module):
    def __init__(self, in_type, out_type, use_act=True, act_fcn='relu'):
        super(EquiLinear, self).__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.output_size = out_type.size
        self.use_act = use_act

        if self.use_act:
            self.activation_function = enn.PointwiseNonLinearity(self.out_type, 'p_' + act_fcn)
        else:
            self.activation_function = None

        if self.output_size != 0:
            self.fc = enn.Linear(self.in_type, self.out_type)
        else:
            self.fc = None

    def forward(self, inputs):
        if self.output_size != 0:
            geo_tensor = self.in_type(inputs, coords)
            geo_tensor_out = self.fc(geo_tensor)
            if self.use_act:
                geo_tensor_out = self.activation_function(geo_tensor_out)
            return geo_tensor_out.tensor
        else:
            return torch.zeros_like(inputs)

class EquiLinearConv(nn.Module):
    def __init__(self, in_type, out_type, use_act=True, act_fcn='relu'):
        super(EquiLinearConv, self).__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.output_size = out_type.size
        self.use_act = use_act

        #self.edge_index = construct_fc_edge_index()

        if self.use_act:
            self.activation_function = enn.PointwiseNonLinearity(self.out_type, 'p_' + act_fcn)
        else:
            self.activation_function = None

        if self.output_size != 0:
            self.fc = enn.R2PointConv(self.in_type, self.out_type, sigma=None, width=2., n_rings=3, frequencies_cutoff=None, bias=True)
        else:
            self.fc = None

    def forward(self, inputs):
        robot_states = robot_states.squeeze()
        robot_states_no_coords = robot_states[:,2:]
        robot_coords = robot_states[:,:2]
        robot_field = self.in_type_robot(robot_states_no_coords, coords=robot_coords)

        #print("ROBOT COORDS: ", robot_coords, type(robot_field))
        #print(robot_field.coords)

        if self.output_size != 0:
            geo_tensor = self.in_type(inputs, coords)
            geo_tensor_out = self.fc(geo_tensor, self.edge_index)
            if self.use_act:
                geo_tensor_out = self.activation_function(geo_tensor_out)
            return geo_tensor_out.tensor
        else:
            return torch.zeros_like(inputs)


class EquiEdgeAttention_M(nn.Module):
    '''
    Class for the robot-human attention module
    '''
    def __init__(self, args, hidden_field, group_helper=None):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EquiEdgeAttention_M, self).__init__()

        self.args = args
        self.group_helper = group_helper

        #self.edge_index = construct_fc_edge_index().to('cuda:0')

        # # Store required sizes
        # self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        # self.human_node_rnn_size = args.human_node_rnn_size
        # self.attention_size = args.attention_size

        # # Linear layer to embed temporal edgeRNN hidden state
        # self.temporal_edge_layer=nn.ModuleList()
        # self.spatial_edge_layer=nn.ModuleList()

        # self.temporal_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))

        # # Linear layer to embed spatial edgeRNN hidden states
        # self.spatial_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))

        grp_act = self.group_helper.grp_act
        scaler = self.group_helper.scaler
        num_rotations = self.group_helper.num_rotations

        #triv for radius
        self.in_type_human = enn.FieldType(
            grp_act,
            self.group_helper.triv_repr
        )
        self.out_type_human = enn.FieldType(
            grp_act,
            NARROW // num_rotations // scaler * self.group_helper.reg_repr
        )
        #print("NUM ROTATIONS: ", num_rotations, 16 // num_rotations // scaler)
        self.human_embedder = EquiLinearConv(self.in_type_human, self.out_type_human, use_act=True, act_fcn='relu')

        # irr for vx vy, irr for gx gy, triv for radius
        self.in_type_robot = enn.FieldType(
            grp_act,
            self.group_helper.irr_repr
            + self.group_helper.irr_repr
            + self.group_helper.triv_repr
        )
        self.out_type_robot = enn.FieldType(
            grp_act,
            NARROW // num_rotations // scaler * self.group_helper.reg_repr
        )

        self.hidden_field = hidden_field

        self.robot_embedder = EquiLinearConv(self.in_type_robot, self.out_type_robot, use_act=True, act_fcn='relu')

        #CHANGE FIRST FIELD TYPE AFTER ADDING MORE PREPROCESSING
        # self.rh_feature_extractor = enn.SequentialModule(
        #     enn.R2PointConv(self.in_type_human, self.hidden_field, sigma=None, width=2., n_rings=3, frequencies_cutoff=None, bias=True),
        #     enn.ReLU(self.hidden_field),
        #     enn.R2PointConv(self.hidden_field, self.hidden_field, sigma=None, width=2., n_rings=3, frequencies_cutoff=None, bias=True),
        #     enn.ReLU(self.hidden_field),
        #     enn.R2PointConv(self.hidden_field, self.hidden_field, sigma=None, width=2., n_rings=3, frequencies_cutoff=None, bias=True),
        #     enn.ReLU(self.hidden_field),
        #     enn.R2PointConv(self.hidden_field, self.hidden_field, sigma=None, width=2., n_rings=3, frequencies_cutoff=None, bias=True),
        #     enn.ReLU(self.hidden_field)
        # )

        self.l1 = enn.R2PointConv(self.in_type_human, self.hidden_field, sigma=None, width=WIDTH, n_rings=N_RINGS, frequencies_cutoff=None, bias=True)
        self.r1 = enn.ReLU(self.hidden_field)
        self.l2 = enn.R2PointConv(self.hidden_field, self.hidden_field, sigma=None, width=WIDTH, n_rings=N_RINGS, frequencies_cutoff=None, bias=True)
        self.r2 = enn.ReLU(self.hidden_field)
        self.l3 = enn.R2PointConv(self.hidden_field, self.hidden_field, sigma=None, width=WIDTH, n_rings=N_RINGS, frequencies_cutoff=None, bias=True)
        self.r3 = enn.ReLU(self.hidden_field)
        self.l4 = enn.R2PointConv(self.hidden_field, self.hidden_field, sigma=None, width=WIDTH, n_rings=N_RINGS, frequencies_cutoff=None, bias=True)
        self.r4 = enn.ReLU(self.hidden_field)

        # number of agents who have spatial edges (complete graph: all 6 agents; incomplete graph: only the robot)
        self.agent_num = 1
        self.num_attention_head = 1

    def forward(self, robot_states, output_spatial, edge_index):
        #robot states (seq_length, nenv, 1, 7) output spatial (seq_length, nenv, human_num, 2)

        #robot_embedding = self.robot_embedder(robot_states)

        #robot_states = robot_states.squeeze()
        #robot_states_no_coords = robot_states[:,2:]
        #robot_coords = robot_states[:,:2]

        #print("SHAPES ", output_spatial.shape, robot_states.shape, edge_index.shape)
        #print("OUTPUT SPATIAL ", output_spatial)
        #print("ROBOT STATES ", robot_states)
        #print("EDGE INDEX ", edge_index)

        all_states = torch.cat((output_spatial, robot_states[:,:,:,:2]), dim=-2)
        #print("ALL STATES SHAPE: ", all_states.shape, all_states)
        all_states_batch = all_states.view(all_states.shape[0] * all_states.shape[1] * all_states.shape[2], all_states.shape[-1])
        #print("all_states_batch shape", all_states_batch.shape, all_states_batch)
        all_radii_batch = torch.full((all_states_batch.shape[0], 1), 0.3).to('cuda:0')
        #all_batch = torch.cat((all_states_batch, all_radii_batch), dim=-1)
        #print("all_states_batch shape", all_radii_batch.shape)

        #robot_field = self.in_type_robot(robot_states_no_coords, coords=robot_coords)

        #print("ROBOT COORDS: ", robot_coords, type(robot_field))
        #print(robot_field.coords)

        all_states_field = self.in_type_human(all_radii_batch, all_states_batch)
        #print("ALL STATES FIELD ", all_states_field.tensor, " COORDS ", all_states_field.coords)
        #all_states_field = self.in_type_human(all_batch, all_states_batch)
        #all_enc = self.rh_feature_extractor(all_states_field, self.edge_index)

        #print("DEVICES: ", all_states_field.tensor.device, edge_index.device, edge_index.shape)

        # #print("EDGE INDEX ", edge_index)
        # l1o = self.l1(all_states_field, edge_index)
        # #print("L1O ", l1o)
        # r1o = self.r1(l1o)
        # #print("R10 ", r1o)
        # l2o = self.l2(r1o, edge_index)
        # r2o = self.r2(l2o)
        # l3o = self.l3(r2o, edge_index)
        # r3o = self.r3(l3o)
        # l4o = self.l4(r1o, edge_index)
        # #print("L4o", l4o)
        # r4o = self.r4(l4o)
        # #print("R4o ", r4o)

        return all_states_field

class EquiActor(nn.Module):
    def __init__(self, args, input_field, hidden_field, group_helper=None):
        super(EquiActor, self).__init__()

        self.args = args
        self.group_helper = group_helper

        #self.edge_index = construct_fc_edge_index().to('cuda:0')

        grp_act = self.group_helper.grp_act
        scaler = self.group_helper.scaler
        num_rotations = self.group_helper.num_rotations

        self.hidden_field = hidden_field
        self.deep_field = enn.FieldType(
            grp_act,
            MID // num_rotations // scaler * self.group_helper.reg_repr
        )
        self.action_field = enn.FieldType(
            grp_act,
            self.group_helper.irr_repr
            + self.group_helper.triv_repr
            + self.group_helper.triv_repr
        )

        self.input_field = input_field

        if DEEP:
            self.inter_conv = enn.R2PointConv(self.input_field, self.deep_field, sigma=None, width=WIDTH, n_rings=N_RINGS, frequencies_cutoff=None, bias=True)
            self.act = enn.ReLU(self.deep_field)
            self.deep_conv1 = enn.R2PointConv(self.deep_field, self.hidden_field, sigma=None, width=WIDTH, n_rings=N_RINGS, frequencies_cutoff=None, bias=True)
            self.deep_act1 = enn.ReLU(self.hidden_field)
            self.deep_conv2 = enn.R2PointConv(self.hidden_field, self.hidden_field, sigma=None, width=WIDTH, n_rings=N_RINGS, frequencies_cutoff=None, bias=True)
            self.deep_act2 = enn.ReLU(self.hidden_field)
        else:
            self.inter_conv = enn.R2PointConv(self.input_field, self.hidden_field, sigma=None, width=WIDTH, n_rings=N_RINGS, frequencies_cutoff=None, bias=True)
            self.act = enn.ReLU(self.hidden_field)
        self.action_conv = enn.R2PointConv(self.hidden_field, self.action_field, sigma=None, width=WIDTH, n_rings=N_RINGS, frequencies_cutoff=None, bias=True)
        init.deltaorthonormal_init(self.action_conv.weights.data, self.action_conv.basissampler)

        self.tanh = nn.Tanh()

    def forward(self, input_state, edge_index):
        ico = self.inter_conv(input_state, edge_index)
        iao = self.act(ico)

        if DEEP:
            dc1 = self.deep_conv1(iao, edge_index)
            da1 = self.deep_act1(dc1)
            dc2 = self.deep_conv2(da1, edge_index)
            da2 = self.deep_act2(dc2)
            action = self.action_conv(da2, edge_index)
        else:
            action = self.action_conv(iao, edge_index)
        return action

class EquiCritic(nn.Module):
    def __init__(self, args, input_field, hidden_field, group_helper=None):
        super(EquiCritic, self).__init__()

        self.args = args
        self.group_helper = group_helper

        grp_act = self.group_helper.grp_act
        scaler = self.group_helper.scaler
        num_rotations = self.group_helper.num_rotations

        self.hidden_field = hidden_field
        self.deep_field = enn.FieldType(
            grp_act,
            MID // num_rotations // scaler * self.group_helper.reg_repr
        )
        self.value_field = enn.FieldType(
            grp_act,
            self.group_helper.triv_repr
        )
        self.value_pool_field = enn.FieldType(
            grp_act,
            self.group_helper.reg_repr
        )

        self.input_field = input_field

        if DEEP:
            self.inter_conv = enn.R2PointConv(self.input_field, self.deep_field, sigma=None, width=WIDTH, n_rings=N_RINGS, frequencies_cutoff=None, bias=True)
            self.act = enn.ReLU(self.deep_field)
            self.deep_conv1 = enn.R2PointConv(self.deep_field, self.hidden_field, sigma=None, width=WIDTH, n_rings=N_RINGS, frequencies_cutoff=None, bias=True)
            self.deep_act1 = enn.ReLU(self.hidden_field)
            self.deep_conv2 = enn.R2PointConv(self.hidden_field, self.hidden_field, sigma=None, width=WIDTH, n_rings=N_RINGS, frequencies_cutoff=None, bias=True)
            self.deep_act2 = enn.ReLU(self.hidden_field)
        else:
            self.inter_conv = enn.R2PointConv(self.input_field, self.hidden_field, sigma=None, width=WIDTH, n_rings=N_RINGS, frequencies_cutoff=None, bias=True)
            self.act = enn.ReLU(self.hidden_field)
        self.value_conv = enn.R2PointConv(self.hidden_field, self.value_field, sigma=None, width=WIDTH, n_rings=N_RINGS, frequencies_cutoff=None, bias=True)
        #self.pool = enn.PointwiseAvgPool2D(self.value_field, kernel_size=21)
        self.pool = enn.GroupPooling(self.value_pool_field)
        self.pool_conv = enn.R2PointConv(self.hidden_field, self.value_pool_field, sigma=None, width=WIDTH, n_rings=N_RINGS, frequencies_cutoff=None, bias=True)

    def forward(self, input_state, edge_index):
        ico = self.inter_conv(input_state, edge_index)
        iao = self.act(ico)

        if DEEP:
            dc1 = self.deep_conv1(iao, edge_index)
            da1 = self.deep_act1(dc1)
            dc2 = self.deep_conv2(da1, edge_index)
            da2 = self.deep_act2(dc2)
            if POOL:
                vp = self.pool_conv(da2, edge_index)
                value = self.pool(vp)
            else:
                value = self.value_conv(da2, edge_index)
        else:
            if POOL:
                vp = self.pool_conv(iao, edge_index)
                #print("VALUE PRE ", iao.tensor.shape, vp.tensor.shape)
                value = self.pool(vp)
                #print("VALUE ", value.tensor.shape)
            else:
                value = self.value_conv(iao, edge_index)
        #("VALUE SHAPEEEEEEEEEEEEEE: ", value.tensor.shape)
        #value_pool = self.pool(value.view(16, 21))
        #print("VALUE PPOOL: ", value_pool.tensor.shape)
        return value

class EquiRobotEncoder(nn.Module):
    def __init__(self, args, hidden_field, group_helper=None):
        super(EquiRobotEncoder, self).__init__()

        self.args = args
        self.group_helper = group_helper

        grp_act = self.group_helper.grp_act
        scaler = self.group_helper.scaler
        num_rotations = self.group_helper.num_rotations

        self.in_type_robot = enn.FieldType(
            grp_act,
            self.group_helper.irr_repr + 
            self.group_helper.triv_repr + 
            self.group_helper.irr_repr + 
            self.group_helper.triv_repr + 
            self.group_helper.triv_repr
        )
        self.hidden_field = hidden_field

        self.conv = enn.R2PointConv(self.in_type_robot, self.hidden_field, sigma=None, width=WIDTH, n_rings=N_RINGS, frequencies_cutoff=None, bias=True)
        self.act = enn.ReLU(self.hidden_field)

    def forward(self, robot_state, edge_index):
        co = self.conv(robot_state, edge_index)
        ao = self.act(co)
        return ao

class EquiHumanEncoder(nn.Module):
    def __init__(self, args, hidden_field, group_helper=None):
        super(EquiHumanEncoder, self).__init__()

        self.args = args
        self.group_helper = group_helper

        grp_act = self.group_helper.grp_act
        scaler = self.group_helper.scaler
        num_rotations = self.group_helper.num_rotations

        self.in_type_human = enn.FieldType(
            grp_act,
            self.group_helper.irr_repr + 
            self.group_helper.triv_repr
        )
        self.hidden_field = hidden_field

        self.conv = enn.R2PointConv(self.in_type_human, self.hidden_field, sigma=None, width=WIDTH, n_rings=N_RINGS, frequencies_cutoff=None, bias=True)
        self.act = enn.ReLU(self.hidden_field)

    def forward(self, human_state, edge_index):
        co = self.conv(human_state, edge_index)
        ao = self.act(co)
        return ao

class equi_SRNN(nn.Module):
    def __init__(self, obs_space_dict, args, infer=False):
        super(equi_SRNN, self).__init__()
        self.infer = infer
        self.is_recurrent = True
        self.args=args

        self.human_num = obs_space_dict['spatial_edges'].shape[0]

        self.seq_length = args.seq_length
        self.nenv = args.num_processes
        self.nminibatch = args.num_mini_batch

        self.disconnected_edge_indices = {}
        self.edge_indices = {}

        # Store required sizes
        self.human_node_rnn_size = args.human_node_rnn_size
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.output_size = args.human_node_output_size

        self.group_helper = GroupHelper(4)
        grp_act = self.group_helper.grp_act
        scaler = self.group_helper.scaler
        num_rotations = self.group_helper.num_rotations
        self.hidden_field = enn.FieldType(
            grp_act,
            WIDE // num_rotations // scaler * self.group_helper.reg_repr
        )
        #print("Hidden field numbers ", num_rotations, scaler, 256 // num_rotations // scaler)
        self.encoding_field = enn.FieldType(
            grp_act,
            MID // num_rotations // scaler * self.group_helper.reg_repr
        )

        self.in_type_human = enn.FieldType(
            grp_act,
            self.group_helper.irr_repr + 
            self.group_helper.triv_repr
        )
        self.in_type_robot = enn.FieldType(
            grp_act,
            self.group_helper.irr_repr + 
            self.group_helper.triv_repr + 
            self.group_helper.irr_repr + 
            self.group_helper.triv_repr + 
            self.group_helper.triv_repr
        )

        #self.equi_rnn = (args)
        self.equi_rh_feature_encoder = EquiEdgeAttention_M(args, self.hidden_field, self.group_helper)
        self.equi_actor = EquiActor(args, self.encoding_field, self.hidden_field, self.group_helper)
        self.equi_critic = EquiCritic(args, self.encoding_field, self.hidden_field, self.group_helper)

        self.equi_human_encoder = EquiHumanEncoder(args, self.encoding_field, self.group_helper)
        self.equi_robot_encoder = EquiRobotEncoder(args, self.encoding_field, self.group_helper)

        self.all_radii = torch.full((1440, 1), 0.3).to('cuda:0')

        if self.args.use_separate_attn:
            self.equi_hh_feature_encoder = EquiSpatialEdgeSelfAttn(args)
            self.equi_rh_feature_encoder = EquiEdgeAttention_M(args)
            #print("WTF ", self.args.use_separate_attn)
        #else:
            #print("NOT SEPARATE")

        # self.actor = actorRNN(
        #     obs_dim,
        #     action_dim,
        #     action_embedding_size,
        #     obs_embedding_size,
        #     rnn_hidden_size,
        #     policy_layers,
        #     rnn_num_layers,
        #     group_helper=group_helper,
        # )

        # self.critic = criticRNN(
        #     obs_dim,
        #     action_dim,
        #     actor_type,
        #     action_embedding_size,
        #     obs_embedding_size,
        #     rnn_hidden_size,
        #     policy_layers,
        #     rnn_num_layers,
        #     group_helper=group_helper
        # )

    def forward(self, inputs, rnn_hxs, masks, infer=False):
        if infer:
            # Test/rollout time
            seq_length = 1
            nenv = self.nenv

        else:
            # Training time
            seq_length = self.seq_length
            nenv = self.nenv // self.nminibatch

        if seq_length in self.edge_indices:
            if nenv in self.edge_indices[seq_length]:
                #print("HIT 1")
                edge_index = self.edge_indices[seq_length][nenv]
            else:
                print("MISS 1")
                self.edge_indices[seq_length][nenv] = construct_fc_edge_index(seq_length, nenv, self.human_num).to('cuda:0')
                edge_index = self.edge_indices[seq_length][nenv]
        else:
            print("MISS 2")
            self.edge_indices[seq_length] = {}
            self.edge_indices[seq_length][nenv] = construct_fc_edge_index(seq_length, nenv, self.human_num).to('cuda:0')
            edge_index = self.edge_indices[seq_length][nenv]

        if seq_length in self.disconnected_edge_indices:
            if nenv in self.disconnected_edge_indices[seq_length]:
                #print("HIT 2")
                disconnected_edge_index_robot = self.disconnected_edge_indices[seq_length][nenv][0]
                disconnected_edge_index_human = self.disconnected_edge_indices[seq_length][nenv][1]
            else:
                print("MISS 3")
                self.disconnected_edge_indices[seq_length][nenv] = [construct_disconnected_edge_index(seq_length, nenv, 0).to('cuda:0'), construct_disconnected_edge_index(seq_length, nenv, self.human_num-1).to('cuda:0')]
                disconnected_edge_index_robot = self.disconnected_edge_indices[seq_length][nenv][0]
                disconnected_edge_index_human = self.disconnected_edge_indices[seq_length][nenv][1]
        else:
            print("MISS 4")
            self.disconnected_edge_indices[seq_length] = {}
            self.disconnected_edge_indices[seq_length][nenv] = [construct_disconnected_edge_index(seq_length, nenv, 0).to('cuda:0'), construct_disconnected_edge_index(seq_length, nenv, self.human_num-1).to('cuda:0')]
            disconnected_edge_index_robot = self.disconnected_edge_indices[seq_length][nenv][0]
            disconnected_edge_index_human = self.disconnected_edge_indices[seq_length][nenv][1]

        # edge_index = construct_fc_edge_index(seq_length, nenv, self.human_num).to('cuda:0')
        # disconnected_edge_index_robot = construct_disconnected_edge_index(seq_length, nenv, 0).to('cuda:0')
        # disconnected_edge_index_human = construct_disconnected_edge_index(seq_length, nenv, self.human_num-1).to('cuda:0')

        robot_node = reshapeT(inputs['robot_node'], seq_length, nenv)
        temporal_edges = reshapeT(inputs['temporal_edges'], seq_length, nenv)
        spatial_edges = reshapeT(inputs['spatial_edges'], seq_length, nenv)

        print("robot node ", robot_node)
        print("spatial edges", spatial_edges)

        # to prevent errors in old models that does not have sort_humans argument
        if not hasattr(self.args, 'sort_humans'):
            self.args.sort_humans = True
        if self.args.sort_humans:
            detected_human_num = inputs['detected_human_num'].squeeze(-1).cpu().int()
        else:
            human_masks = reshapeT(inputs['visible_masks'], seq_length, nenv).float() # [seq_len, nenv, max_human_num]
            # if no human is detected (human_masks are all False, set the first human to True)
            human_masks[human_masks.sum(dim=-1)==0] = self.dummy_human_mask


        hidden_states_node_RNNs = reshapeT(rnn_hxs['human_node_rnn'], 1, nenv)
        masks = reshapeT(masks, seq_length, nenv)

        if self.args.no_cuda:
            all_hidden_states_edge_RNNs = Variable(
                torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cpu())
        else:
            all_hidden_states_edge_RNNs = Variable(
                torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cuda())
            all_hidden_states_node_RNNs = Variable(
                torch.zeros(1, nenv, 1, rnn_hxs['human_node_rnn'].size()[-1]).cuda())
            #print("ALL HIDDEN STATES EDGE ", all_hidden_states_edge_RNNs.shape, all_hidden_states_node_RNNs.shape)

        # spatial_attn_out_int = self.equi_hh_feature_encoder(spatial_edges, detected_human_num)
        # spatial_attn_out = spatial_attn_out_int.view(seq_length, nenv, self.human_num, -1)

        #output_spatial = self.equi_spatial_linear(spatial_attn_out)
        output_spatial = spatial_edges
        #print("Output spatial and robot node ", output_spatial.shape, robot_node[0,0,0,:])

        print("edge indices:")
        print(edge_index)
        print("robot")
        print(disconnected_edge_index_robot)
        print("human")
        print(disconnected_edge_index_human)

        all_states = torch.cat((output_spatial[:,:,:,:2], robot_node[:,:,:,:2]), dim=-2)
        #print("ALL STATES SHAPE: ", all_states.shape, all_states)
        all_states_batch = all_states.view(all_states.shape[0] * all_states.shape[1] * all_states.shape[2], all_states.shape[-1])
        print("all_states_batch shape", all_states_batch.shape, all_states_batch)
        #all_radii_batch = torch.full((all_states_batch.shape[0], 1), 0.3).to('cuda:0')
        #print("all radii batch ", all_radii_batch.shape)
        #all_radii_batch = self.all_radii[:all_states_batch.shape[0]]
        #all_batch = torch.cat((all_states_batch, all_radii_batch), dim=-1)
        #print("all_states_batch shape", all_radii_batch.shape)

        #robot_field = self.in_type_robot(robot_states_no_coords, coords=robot_coords)

        #print("ROBOT COORDS: ", robot_coords, type(robot_field))
        #print(robot_field.coords)

        robot_node_batch = robot_node.view(robot_node.shape[0] * robot_node.shape[1] * robot_node.shape[2], robot_node.shape[-1])
        robot_state_input = self.in_type_robot(robot_node_batch[:,2:], robot_node_batch[:,:2])
        print("robot node batch ", robot_node_batch.shape, robot_node_batch)
        print("robot state input ", robot_state_input.shape, robot_state_input)

        output_spatial_batch = output_spatial.view(output_spatial.shape[0] * output_spatial.shape[1] * output_spatial.shape[2], output_spatial.shape[-1])
        #self.human_radii_batch = torch.full((output_spatial_batch.shape[0], 1), 0.3).to('cuda:0')
        #human_radii_batch = self.all_radii[:output_spatial_batch.shape[0]]
        #print("human radii batch", human_radii_batch.shape)
        human_state_input = self.in_type_human(output_spatial_batch[:,2:], output_spatial_batch[:,:2])
        #print("output spatial batch ", output_spatial_batch.shape, output_spatial_batch)
        #print("human state input ", human_state_input.shape, human_state_input)

        #combined_coords = torch.cat((output_spatial_batch, robot_node_batch[:,:,:,:2]), dim=-2)

        #print("Robot human node batch shapes ", robot_node_batch.shape, output_spatial_batch.shape)

        #print("SEQ LENGTH ", seq_length, nenv)
        #print("EDGE INDEX SHAPE FOR REAL ", edge_index.shape)

        #rh_encoding = self.in_type_human(all_radii_batch, all_states_batch)

        robot_encoding = self.equi_robot_encoder(robot_state_input, disconnected_edge_index_robot)
        human_encoding = self.equi_human_encoder(human_state_input, disconnected_edge_index_human)

        print("robot encoding")
        print(robot_encoding)
        print("human encoding")
        print(human_encoding)

        #print("Robot and human encodings ", robot_encoding.tensor.shape, human_encoding.tensor.shape)

        robot_encoding_batch = robot_encoding.tensor.view(robot_node.shape[0], robot_node.shape[1], robot_node.shape[2], robot_encoding.shape[-1])
        output_encoding_batch = human_encoding.tensor.view(output_spatial.shape[0], output_spatial.shape[1], output_spatial.shape[2], human_encoding.shape[-1])
        combined_encoding_batch = torch.cat((output_encoding_batch, robot_encoding_batch), dim=-2)

        print("robot encoding batch")
        print(robot_encoding_batch)
        print("output encoding batch")
        print(output_encoding_batch)
        print("combined encoding batch")
        print(combined_encoding_batch)

        #print("Robot and human encoding batch ", robot_encoding_batch.shape, output_encoding_batch.shape, combined_encoding_batch.shape)

        combined_encoding = combined_encoding_batch.view(combined_encoding_batch.shape[0] * combined_encoding_batch.shape[1] * combined_encoding_batch.shape[2], combined_encoding_batch.shape[-1])
        print("combined encoding")
        print(combined_encoding)
        print("combined encoding ", combined_encoding.shape)

        rh_encoding = self.encoding_field(combined_encoding, all_states_batch)

        #rh_encoding = self.equi_rh_feature_encoder(robot_node, output_spatial, edge_index)
        #print("RH ENCODING SHAPE: ", rh_encoding.tensor.shape)

        #outputs, h_nodes \
        #    = self.equi_rnn(robot_states, hidden_attn_weighted, hidden_states_node_RNNs, masks)

        #all_hidden_states_node_RNNs = h_nodes
        #outputs_return = outputs

        rnn_hxs['human_node_rnn'] = all_hidden_states_node_RNNs
        rnn_hxs['human_human_edge_rnn'] = all_hidden_states_edge_RNNs

        #x = outputs_return[:,:,0,:]
        hidden_critic = self.equi_critic(rh_encoding, edge_index)
        hidden_actor = self.equi_actor(rh_encoding, edge_index)

        print("hidden actor shape ", hidden_actor.shape, hidden_actor)

        action_indices = torch.arange(start=self.human_num, end=(seq_length * nenv * (self.human_num+1)), step=(self.human_num+1)).to('cuda:0')
        print(action_indices)
        robot_actions = hidden_actor[action_indices]
        print("robot actions ", robot_actions)
        robot_values = hidden_critic[action_indices]

        #print("ROBOT ACTIONS SHAPE ", robot_actions.shape, robot_values.shape)

        for key in rnn_hxs:
            rnn_hxs[key] = rnn_hxs[key].squeeze(0)

        #if infer:
            #cl = self.equi_critic_linear(hidden_critic).squeeze(0)
            #cl = hidden_critic.squeeze(0)
        return robot_values.tensor, robot_actions.tensor, rnn_hxs