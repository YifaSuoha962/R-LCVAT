import os
import sys
import os.path as osp
import time

sys.path.insert(0, osp.dirname(os.path.abspath(__file__)))

from rdkit import Chem
import numpy as np
import torch

from models.graph_rel_transformers import Graph_Transformer_Base
from models.graph_vae_transformers import Graph_Transformer_VAE
from models.seq_rel_transformers import Seq_Transformer_Base
from models.seq_vae_transformers import Seq_Transformer_VAE

from models.module_utils import Model_Save, eval_plot, coverage_plot, beam_result_process, process_multi_rxn_coverage
from utils.preprocess_smi import canonicalize_smiles
from utils.parsing import get_parser, post_setting_args
from utils.chem_tools import NODE_FDIM, BOND_FDIM
from utils.wrap_single_smi import InferBatch, InferBatch_wo_rxns


def BiG2S_Inference(args, input_smi, rxn_type):
    Infer_wrapper = InferBatch_wo_rxns('./preprocessed', args)
    batch_graph_input = Infer_wrapper.preprocess(input_smi, rxn_type)
    if args.use_reaction_type:
        dec_cls = 2
    else:
        dec_cls = 1

    ckpt_dir = os.path.join('checkpoints', args.save_name)
    token_idx = Infer_wrapper.token_index
    module_saver = Model_Save(
        ckpt_dir=ckpt_dir,
        device=args.device,
        save_strategy=args.save_strategy,
        save_num=args.save_num,
        swa_count=args.swa_count,
        swa_tgt=args.swa_tgt,
        const_save_epoch=args.const_save_epoch,
        top1_weight=args.top1_weight
    )

    if args.model_type == 'BiG2S':
        predict_module = Graph_Transformer_Base(
            f_vocab=len(token_idx),
            f_atom=NODE_FDIM + 10 if args.use_reaction_type else NODE_FDIM,
            f_bond=BOND_FDIM,
            token_idx=Infer_wrapper.token_index,
            token_freq=None,
            token_count=None,
            cls_len=dec_cls,
            args=args
        )
    elif args.model_type == 'BiG2S_HCVAE':
        predict_module = Graph_Transformer_VAE(
            f_vocab=len(token_idx),
            f_atom=NODE_FDIM + 10 if args.use_reaction_type else NODE_FDIM,
            f_bond=BOND_FDIM,
            token_idx=Infer_wrapper.token_index,
            token_freq=None,
            token_count=None,
            cls_len=dec_cls,
            args=args
        )
    elif args.model_type == 'S2S_HCVAE':
        predict_module = Seq_Transformer_VAE(
            f_vocab=len(token_idx),
            token_idx=Infer_wrapper.token_index,
            token_freq=None,
            token_count=None,
            cls_len=dec_cls,
            args=args
        )
    else:  # pure transformer : S2S
        predict_module = Seq_Transformer_Base(
            f_vocab=len(token_idx),
            token_idx=Infer_wrapper.token_index,
            token_freq=None,
            token_count=None,
            cls_len=dec_cls,
            args=args
        )

    _, predict_module.model, _, _ = module_saver.load(args.ckpt_list[0], predict_module.model)

    # ckpt_file = osp.join(pretrained_path, 'swa9.ckpt')
    # module_ckpt = torch.load(ckpt_file, map_location=args.device)
    # predict_module.model.load_state_dict(module_ckpt['module_param'])

    with torch.no_grad():
        batch_data = batch_graph_input.to(args.device)
        predict_result, predict_scores, rxn_types, prob_per_token = predict_module.model_predict(
            data=batch_data,
            args=args
        )       # TODO: add rxn_type in prediction

    smi_nodes_sorted, prob_nodes_sorted = Infer_wrapper.post_process(predict_result, predict_scores)

    return smi_nodes_sorted, predict_scores, prob_per_token


if __name__ == '__main__':

    parser = get_parser(mode='test')
    args = post_setting_args(parser)
    args.use_reaction_type = False
    args.beam_size = 10

    # 需要手动更改的地方： 1. dataset_name, 2. chekpoint_path, 3. 产物分子
    # 用哪个数据集的 checkpoint 和表，就写成哪个

    # 调整 beam search 里的 温度系数
    args.T = 1.2
    assert args.dataset_name in ['debug_uspto_50k', 'pistachio', 'uspto_diverse', 'uspto_50k_infer']
    assert args.T in [0.7, 1.2, 1.6]

    if args.use_subs and (args.use_reaction_type or args.model_type == 'BiG2S_HCVAE'):
        dec_cls = 2
    elif args.use_subs or args.use_reaction_type or args.model_type == 'BiG2S_HCVAE':
        dec_cls = 1
    else:
        dec_cls = 0

    #***************************#
    # input_smi = 'N # C c 1 n n ( - c 2 c ( C l ) c c ( C ( F ) ( F ) F ) c c 2 C l ) c c 1 C ( B r ) = C ( C l ) B r'.replace(' ', '')  # demo_a
    # input_smi = 'O C ( c 1 c c c c c 1 ) ( c 1 c c c c c 1 ) c 1 c c c c c 1 Cl'.replace(' ', '')  # demo of pistachio
    # input_smi = 'C c 1 n c ( C # N ) c c c 1 Br'.replace(' ', '')  # demo of pistachio
    # input_smi = "C N c 1 c c c c ( N ) c 1 C # N".replace(' ', '')

    input_smi = "O=[N+]([O-])c1ccc(F)cc1CBr"
    # input_smi = "CC(C)N1C(=O)CC(C)(C)c2cc(N)ccc21"

    # ans_1 = 'Cl c 1 c c c c c 1 Br . O = C ( c 1 c c c c c 1 ) c 1 c c c c c 1'.replace(' ', '')
    # ans_2 = '[Li] c 1 c c c c c 1 .    O = C ( c 1 c c c c c 1 ) c 1 c c c c c 1 Cl'.replace(' ', '')

    # refs = ['Cl c 1 c c c c c 1 Br . O = C ( c 1 c c c c c 1 ) c 1 c c c c c 1'.replace(' ', ''),
    #        '[Li] c 1 c c c c c 1 .    O = C ( c 1 c c c c c 1 ) c 1 c c c c c 1 Cl'.replace(' ', '')]

    # refs = ["C C # N . C c 1 n c ( I ) c c c 1 Br".replace(' ', ''),
    #         "C c 1 n c ( F ) c c c 1 Br . [C-] # N".replace(' ', ''),
    #         "C c 1 n c ( Br ) c c c 1 Br . N # C [Cu]".replace(' ', ''),
    #         "C c 1 n c ( I ) c c c 1 Br . N # C [Cu] C # N".replace(' ', '')]

    refs = ['C N . N # C c 1 c ( N ) c c c c 1 F'.replace(' ', ''),
            'C [NH3+] . N # C c 1 c ( N ) c c c c 1 F'.replace(' ', ''),
            'C N c 1 c c c c ( [N+] ( = O ) [O-] ) c 1 C # N'.replace(' ', '')]

    # 为了方便，用数字代表反应类型序号
    rxn_type = 5
    #***************************#

    start = time.perf_counter()
    top_k, tot_score, token_score = BiG2S_Inference(args, input_smi, rxn_type)

    filtered_top_k = [smi.split(',')[1] for smi in top_k[0]]
    max_len = max([len(s) for s in filtered_top_k])

    print(f"top-1 correct?: {top_k[0][0].split(',')[1] in refs}")
    print(f"top-1 correct?: {top_k[0][1].split(',')[1] in refs}\n")

    end = time.perf_counter()
    print(top_k)
    print(tot_score)
    print('推理时间: %s 秒' % (end - start))

