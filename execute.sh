rsync -av --exclude='.venv' --exclude='wandb' --exclude='.git' --exclude='results/ppo**' --exclude='results/**/checkpoint*' ./ ~/Tiny_Reasoner/

ssh "$1" << """
    mkdir -p /Data/joao.giordani-donasolo/
    yes | cp -r Tiny_Reasoner/ /Data/joao.giordani-donasolo/
    cd /Data/joao.giordani-donasolo/Tiny_Reasoner/
    rm -f nohup.out
    nohup uv run python -u -m experiments.ppo.train --config configs/ppo_format_shaped_pretrained.yaml &
    disown
"""