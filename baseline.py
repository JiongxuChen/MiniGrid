"""Portable training, evaluation, validation-only selection and result summary."""
import argparse,csv,json
from pathlib import Path
from minigrid_baseline.config import ROOT,load_episode_seeds,select_seeds
from minigrid_baseline.io import write_csv,write_json,file_hash
from minigrid_baseline.metrics import episode_metrics

def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def best(run):
    candidates=[r for r in rows(run/'learning_curve.csv') if int(r['checkpoint_step'])>0]
    r=min(candidates,key=lambda r:(-float(r['success_rate']),-float(r['mean_return']),int(r['checkpoint_step'])))
    step=int(r['checkpoint_step'])
    return (run/'final_model.zip' if step==1000000 else run/f'checkpoints/step_{step:09d}.zip'),step

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['evaluate','best','summarize','plot','verify']);p.add_argument('--root',type=Path,default=ROOT/'results');p.add_argument('--device',default='cpu');p.add_argument('--report-dir',type=Path);a=p.parse_args()
    root=a.root.resolve()
    report=a.report_dir or (ROOT/'reports' if root==ROOT/'results' else root/'summary')
    if a.stage in ('evaluate','best'):
        import torch
        from stable_baselines3 import PPO
        from minigrid_baseline.evaluation import evaluate,save_evaluation
        torch.set_num_threads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        data=load_episode_seeds()
        for seed in range(3):
            run=root/f'seed_{seed}'
            path,step=best(run) if a.stage=='best' else (run/'final_model.zip',1000000)
            model=PPO.load(path,device=a.device)
            if a.stage=='evaluate':assert model.num_timesteps==model.last_update_step==1000000
            for size in (6,8):
                env=f'MiniGrid-DoorKey-{size}x{size}-v0';folder=run/('best_test' if a.stage=='best' else 'final_test')/env
                if folder.exists():raise FileExistsError(f'{folder} exists; use a fresh run root to reproduce results')
                folder.mkdir(parents=True)
                rs,m=evaluate(model,env,select_seeds(data,env,'test'),seed,step,'test',path)
                save_evaluation(rs,m,folder/'episodes.csv',folder/'summary.json',path)
    elif a.stage=='summarize':
        import statistics
        per=[]
        for seed in range(3):
            for kind in ('final_test','best_test'):
                for size in (6,8):
                    path=root/f'seed_{seed}/{kind}/MiniGrid-DoorKey-{size}x{size}-v0/episodes.csv'
                    per.append(dict(seed=seed,selection='final' if kind=='final_test' else 'best',size=size,**episode_metrics(rows(path))))
        folder=report;folder.mkdir(parents=True,exist_ok=True);write_csv(folder/'per_seed.csv',per)
        groups=[]
        for selection in ('final','best'):
            for size in (6,8):
                rates=[r['success_rate'] for r in per if r['selection']==selection and r['size']==size]
                groups.append(dict(selection=selection,size=size,mean=statistics.mean(rates),sample_sd=statistics.stdev(rates)))
        write_csv(folder/'across_seeds.csv',groups);print(json.dumps(groups,indent=2))
    elif a.stage=='plot':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),sharey=True)
        for seed,ax in enumerate(axes):
            ps=rows(root/f'seed_{seed}/learning_curve.csv');ax.plot([int(r['checkpoint_step'])/1e6 for r in ps],[float(r['success_rate'])*100 for r in ps]);ax.set(title=f'Seed {seed}',xlabel='Training actions (million)',ylabel='Validation success (%)',ylim=(0,100));ax.grid(alpha=.2)
        folder=report;folder.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(folder/'learning_curves.png',dpi=160);fig.savefig(folder/'learning_curves.pdf')
    else:
        manifest=json.loads((ROOT/'ARTIFACTS.json').read_text())
        for name,digest in manifest['files'].items():assert file_hash(ROOT/name)==digest,name
        for seed in range(3):
            run=root/f'seed_{seed}';chosen,step=best(run)
            for kind,model in [('final_test',run/'final_model.zip'),('best_test',chosen)]:
                for size in (6,8):
                    rs=rows(run/f'{kind}/MiniGrid-DoorKey-{size}x{size}-v0/episodes.csv');assert len(rs)==100
                    assert [int(r['episode_seed']) for r in rs]==load_episode_seeds()[f'test_{size}x{size}']
                    assert all(r['model_file_sha256']==file_hash(model) for r in rs)
        print('Artifact hashes, fixed test lists and model references verified')
if __name__=='__main__':main()
