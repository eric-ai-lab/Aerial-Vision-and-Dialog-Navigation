import os
import json
import time
import numpy as np
from collections import defaultdict

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    #先加入绝对路径，否则会报错，注意__file__表示的是当前执行文件的路径
from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results



from xview_et.agent import NavCMTAgent
from env import ANDHNavBatch
from xview_et.parser import parse_args

def get_tokenizer(args):
    from transformers import AutoTokenizer
    cfg_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def build_dataset(args, rank=0, is_test=False):
    tok = get_tokenizer(args)

    dataset_class = ANDHNavBatch

    train_env = dataset_class(
        args.train_anno_dir, args.train_dataset_dir, ['train'], 
        tokenizer=tok, 
        max_instr_len=args.max_instr_len,
        batch_size=args.batch_size, 
        seed=args.seed+rank,
        full_traj = False,
    )
    train_full_traj_env = None

    val_env_names = ['val_seen', 'val_unseen',  ] #'test_unseen' 

    if args.submit:
        val_env_names.append('test_unseen')
        
    val_envs = {}
    for split in val_env_names:


        val_env = dataset_class(
            args.val_anno_dir, args.val_dataset_dir, [split], 
            tokenizer=tok, 
            max_instr_len=args.max_instr_len,
            batch_size=args.batch_size, 
            seed=args.seed+rank,
            full_traj = False,
        )

        val_envs[split] = val_env

    val_full_traj_envs = None

    return train_env, train_full_traj_env, val_envs, val_full_traj_envs

def train(args, train_env, train_full_traj_env, val_envs, val_full_traj_envs, aug_env=None, rank=-1):
    default_gpu = is_default_gpu(args)

    
    with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
        json.dump(vars(args), outf, indent=4)
    writer = SummaryWriter(log_dir=args.log_dir)
    record_file = os.path.join(args.log_dir, 'train.txt')
    write_to_record_file(str(args) + '\n\n', record_file)

    agent_class = NavCMTAgent
    agent = agent_class(args, rank=rank)

    # resume file
    start_iter = 0
    if args.resume_file is not None:
        start_iter = agent.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration {}".format(args.resume_file, start_iter),
                record_file
            )
       
    # first evaluation
    if args.eval_first:
        loss_str = "validation before training"
        if args.train_val_on_full:
            # evaluate in full traj mode
            for env_name, env in val_full_traj_envs.items():
                env_name += '_full_traj'
                agent.env = env

                loader = torch.utils.data.DataLoader(env, batch_size = 1)
                agent.test(loader, feedback='student')
                preds = agent.get_results()


                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], 0)

        else:
            for env_name, env in val_envs.items():
                agent.env = env
                loader = torch.utils.data.DataLoader(env, batch_size = 1)
                # Get validation distance from goal under test evaluation conditions
                agent.test(loader, feedback='student')
                pred_results = agent.get_results()

                score_summary, result = env.eval_metrics(pred_results)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], 0)
       
        for env_name, env in val_envs.items():
            env_name += '_human_att'
            agent.env = env 

            loader = torch.utils.data.DataLoader(env, batch_size = 1)
            agent.test(loader, feedback='teacher') # use teacher mode to evaluate human attention pred
            preds = agent.get_results()


            score_summary, _ = env.eval_metrics(preds, human_att_eval = True)
            loss_str += ", %s " % env_name
            for metric, val in score_summary.items():
                loss_str += ', %s: %.2f' % (metric, val)
                writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], 0)

        write_to_record_file(loss_str, record_file)


    # Start Training
    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )
    
    best_val = {'val_unseen': {"spl": 0., "state":""}, 'val_unseen_full_traj': {"spl": 0., "state":""}}

    interval = int(train_env.size()/args.batch_size) * args.log_every

    zero_start_iter = 0
    for idx in range(start_iter, start_iter+args.iters, interval):
        agent.logs = defaultdict(list)
        
        iter = idx + interval
        if args.train_val_on_full:
            agent.env = train_full_traj_env
        else:
            agent.env = train_env
        loader = torch.utils.data.DataLoader(agent.env, batch_size = 1)

        # Train for 2 epochs before evaluate again
        agent.train(loader, args.log_every, feedback=args.feedback, nss_w_weighting = 1 ) # nss_w_weighting = max(0, (args.iters/2 - idx)/ (args.iters/2)))

        IL_loss = sum(agent.logs['IL_loss']) / max(len(agent.logs['IL_loss']), 1)

        writer.add_scalar("loss/IL_loss", IL_loss, iter)

        write_to_record_file(
            "\nIL_loss %.4f" % (
                IL_loss),
            record_file
        )

        # Run validation
        loss_str = "iter {}".format(iter)
        
        agent.save(iter, os.path.join(args.ckpt_dir, "latest_dict_" + str(iter)))
        agent_class_eval = NavCMTAgent
        agent_eval = agent_class_eval(args, rank=rank)
        print("Loaded the listener model at iter %d from %s" % \
            (agent_eval.load(os.path.join(args.ckpt_dir, "latest_dict_" + str(iter))),
                os.path.join(args.ckpt_dir, "latest_dict_" + str(iter)))\
            )


        if args.train_val_on_full:
            # evaluate in full traj mode
            for env_name, env in val_full_traj_envs.items():
                env_name += '_full_traj'
                agent_eval.env = env

                loader = torch.utils.data.DataLoader(env, batch_size = 1)
                agent_eval.test(loader, feedback='student')
                preds = agent_eval.get_results()


                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], iter)
                if env_name in best_val:
                    if score_summary['spl'] >= best_val[env_name]['spl']:
                        best_val[env_name]['spl'] = score_summary['spl']
                        best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                        agent_eval.save(iter, os.path.join(args.ckpt_dir, "best_%s" % (env_name)))

        else:
            for env_name, env in val_envs.items():
                agent_eval.env = env
                loader = torch.utils.data.DataLoader(env, batch_size = 1)
                # Get validation distance from goal under test evaluation conditions
                agent_eval.test(loader, feedback='student')
                pred_results = agent_eval.get_results()

                score_summary, result = env.eval_metrics(pred_results)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], iter)
                if env_name in best_val:
                    if score_summary['spl'] >= best_val[env_name]['spl']:
                        best_val[env_name]['spl'] = score_summary['spl']
                        best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                        agent_eval.save(iter, os.path.join(args.ckpt_dir, "best_%s" % (env_name)))

        
        
        # # evaluate in full traj mode
        # for env_name, env in val_full_traj_envs.items():
        #     env_name += '_full_traj'
        #     agent.env = env

        #     loader = torch.utils.data.DataLoader(env, batch_size = 1)
        #     agent.test_full_traj(loader, feedback='student')
        #     preds = agent.get_results()


        #     score_summary, _ = env.eval_metrics(preds)
        #     loss_str += ", %s " % env_name
        #     for metric, val in score_summary.items():
        #         loss_str += ', %s: %.2f' % (metric, val)
        #         writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], iter)
        #     # select model by gp
       
        # evaluate human attention
        for env_name, env in val_envs.items():
            env_name += '_human_att'
            agent_eval.env = env

            loader = torch.utils.data.DataLoader(env, batch_size = 1)
            agent_eval.test(loader, feedback='teacher') # use teacher mode to evaluate human attention pred
            preds = agent_eval.get_results()


            score_summary, _ = env.eval_metrics(preds, human_att_eval = True)
            loss_str += ", %s " % env_name
            for metric, val in score_summary.items():
                loss_str += ', %s: %.2f' % (metric, val)
                writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], iter)

        

        write_to_record_file(
            ('%s (%d %d%%) %s' % (timeSince(start, float(iter)/args.iters), iter, float(iter)/args.iters*100, loss_str)),
            record_file
        )
        write_to_record_file("BEST RESULT TILL NOW", record_file)
        for env_name in best_val:
            write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)
        zero_start_iter += interval


def valid(args, val_envs,  val_full_traj_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    agent_class = NavCMTAgent
    agent = agent_class(args, rank=rank)
    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (agent.load(args.resume_file), args.resume_file))

    with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
        json.dump(vars(args), outf, indent=4)
    record_file = os.path.join(args.log_dir, 'valid.txt')
    write_to_record_file(str(args) + '\n\n', record_file)
    loss_str = "iter {}".format(iter)

    if args.train_val_on_full:
        # evaluate in full traj mode
        for env_name, env in val_full_traj_envs.items():
            env_name += '_full_traj'
            agent.env = env

            loader = torch.utils.data.DataLoader(env, batch_size = 1)
            agent.test(loader, feedback='student')
            preds = agent.get_results()

            score_summary, result = env.eval_metrics(preds)
            loss_str += "Env name: %s" % env_name
            for metric, val in score_summary.items():
                loss_str += ', %s: %.2f' % (metric, val)
            write_to_record_file(loss_str+'\n', record_file)
            json.dump(
                result,
                open(os.path.join(args.pred_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )

    else:
        for env_name, env in val_envs.items():
            agent.env = env
            loader = torch.utils.data.DataLoader(env, batch_size = 1)
            # Get validation distance from goal under test evaluation conditions
            agent.test(loader, feedback='student')
            pred_results = agent.get_results()

            if env_name == 'test_unseen':
                np.save('./output_test_result.npy', pred_results)
            else:
                score_summary, result = env.eval_metrics(pred_results)
                loss_str += "Env name: %s" % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                write_to_record_file(loss_str+'\n', record_file)
                json.dump(
                    result,
                    open(os.path.join(args.pred_dir, "eval_detail_%s.json" % env_name), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )


def main():
    args = parse_args()
    if args.train_val_on_full:
        args.max_action_len *= 4
    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0
    if args.vision_only:
        print("!!! Vision only")
    if args.language_only:
        print("!!! Language only")

    set_random_seed(args.seed + rank)
    train_env, train_full_traj_env, val_envs, val_full_traj_envs = build_dataset(args, rank=rank)

    if not args.test:
        train(args, train_env, train_full_traj_env, val_envs, val_full_traj_envs, rank=rank)
    else:
        valid(args, val_envs, val_full_traj_envs, rank=rank)
            

if __name__ == '__main__':
    main()
