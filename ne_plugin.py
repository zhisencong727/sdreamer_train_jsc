# get ne data

import subprocess
import sys
from data_provider.data_generator_ne import data_generator
import argparse
import torch
from layers.transformer import SWTransformer

activation = "glu"
norm_type = "layernorm"
patch_len = 16
seed = 42
ca_layers = 1
batch_size = 64
n_sequences = 64
ne_patch_len = 32
e_layers = 2
fold = 1

# BaseLine_Seq_pl16_el2_cl1_f1_seql3_kl_2.0_t3.5

config = dict(
    seed=42,
    is_training=1,
    model_id="test",
    model="BaseLine",
    data="Seq",
    isNE=True,
    fold=1,
    root_path="",
    data_path="processedTrainingData/",
    features="ALL",
    n_sequences=n_sequences,
    useNorm=True,
    num_workers=10,
    seq_len=512,
    patch_len=patch_len,
    ne_patch_len=ne_patch_len,
    stride=8,
    padding_patch="end",
    subtract_last=0,
    decomposition=0,
    kernel_size=25,
    individual=0,
    mix_type=0,
    c_out=3,
    d_model=128,
    n_heads=8,
    e_layers=e_layers,
    ca_layers=ca_layers,
    seq_layers=3,
    d_ff=512,
    dropout=0.1,
    path_drop=0.0,
    pos_emb="learned",
    activation=activation,
    norm_type=norm_type,
    output_attentions=False,
    useRaw=False,
    epochs=100,
    batch_size=batch_size,
    patience=30,
    optimizer="adamw",
    lr=0.001,
    weight_decay=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    eps=1e-9,
    scheduler="CosineLR",
    scale=0.0,
    pct_start=0.3,
    step_size=30,
    gamma=0.5,
    weight=[1, 1, 1],
    visualize_mode=[],
    visualizations="",
    # checkpoints=checkpoints,
    reload_best=True,
    reload_ckpt=None,
    use_gpu=True,
    gpu=0,
    use_multi_gpu=False,
    test_flop=False,
    print_freq=50,
    # output_path=output_path,
    # ne_patch_len=ne_patch_len,
    # des=des_name,
    # pad=False,
)

parser = argparse.ArgumentParser(description="Transformer family for sleep scoring")
args = parser.parse_args()
parser_dict = vars(args)
for k, v in config.items():
    parser_dict[k] = v

c_in = 1
c_out = args.c_out
d_model = args.d_model
n_heads = args.n_heads
seq_len = args.seq_len
dropout = args.dropout
path_drop = args.path_drop
e_layers = args.e_layers
patch_len = args.patch_len
ne_patch_len = args.ne_patch_len
norm_type = args.norm_type
activation = args.activation
n_sequences = args.n_sequences
output_attentions = args.output_attentions
d_head = d_model // n_heads
inner_dim = n_heads * d_head
mult_ff = args.d_ff // d_model
n_traces = 3 if args.features == "ALL" else 1

assert (seq_len % patch_len) == 0
n_patches = seq_len // patch_len

### seq_len for ne is 512 as well?
n_patches_ne = (seq_len // ne_patch_len)+1

ne_transformer = SWTransformer(
    ne_patch_len,
    n_patches_ne,
    e_layers,
    c_in,
    inner_dim,
    n_heads=n_heads,
    d_head=d_head,
    dropout=dropout,
    path_drop=path_drop,
    activation=activation,
    norm=norm_type,
    mult=mult_ff,
    mix_type=args.mix_type,
    cls=True,
    flag="seq",
    domain="time",
    output_attentions=output_attentions,
    stride=ne_patch_len,
    pad=True,
        )

def get_data(args,flag):
    data_set,data_loader = data_generator(args,flag)
    return data_set,data_loader

if __name__ == "__main__":
    # write the necessary training files
    print("RUNNING")
    subprocess.run([sys.executable, "write_training_data_ne.py"])
    print("RAN")
    # load ne using data loader
    data_set,data_loader = get_data(args,"train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, (traces, nes, labels) in enumerate(data_loader):
        nes = nes.to(device)

    # plug it in to the ne transformer
        ne_out, ne_attn = ne_transformer(nes)
        print(i)
        print(ne_out.shape)
    

