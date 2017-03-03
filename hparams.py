from model import policies

"""Single place to store definition of new hyperparameters"""


def add_hparams(parser):
    "Adds hyperparameters to an argparse.ArgumentParser object"
    parser.add_argument('--clearance', default=6, type=int, help="PongCatastropheWrapper")
    parser.add_argument('--location', default="bottom", choices=["top", "bottom"], help="PongCatastropheWrapper")
    parser.add_argument('--learning_rate', default=0.0001, type=float, help="A3C")
    parser.add_argument("--record_interval", default=0,
                        help="FrameSaverWrapper: How frequently to record episodes. 0 = never, 1 = every frame")
    parser.add_argument("--max_episodes", default=0, type=int,
                        help="A3C: Maximum number of episodes (per worker) to record, 0 = no limit")
    parser.add_argument("--classifier_file", default = "", help="tensorflow checkpoint storint catastrophe classifier")
    parser.add_argument("--blocker_file", default = "", help="pickle file storing blocker")
    parser.add_argument("--allowed_actions_source", default = "blocker",
                        choices=["blocker", "heuristic"], help="Source of allowed actions")
    parser.add_argument("--blocking_mode", default = "none",
                        choices=["none","action_pruning", "action_replacement"], help="Method of blocking")
    parser.add_argument("--reward_scale", default=1.0, type=float, help="Non-catatrophe rewards are divided by this amount")
    parser.add_argument("--catastrophe_type", default="1", type=str, help="Catastrophe type for pong")
    
    # exploration
    parser.add_argument("--prior_count", type=float, default=100, help="prior state visitation counts")

    parser.add_argument("--lstm_size", type=int, default=256, help="lstm state size")
    parser.add_argument("--entropy_scale", type=float, default=0.01, help="scale of the entropy bonus")
    parser.add_argument("--clip_norm", type=float, default=40.0, help="global variable norm at which to clip gradients")
    parser.add_argument("--local_steps", type=int, default=20, help="number of steps between policy gradient updates")
    parser.add_argument("--policy", type=str, default="lstm", choices=policies.keys(), help="policy type")
    parser.add_argument("--death_penalty", type=float, default=0, help="Impose penalty for dying. Relevant to Pacman, Montezuma.")

    parser.add_argument("--deterministic", type=int, default=1, help="Use deterministic environment.")
    return parser

def get_hparams(args):
    """Returns dictionary of hyperparameters from parsed arguments"""
    return {
        "clearance": args.clearance,
        "location": args.location,
        "learning_rate": args.learning_rate,
        "record_interval": args.record_interval,
        "max_episodes": args.max_episodes,
        "classifier_file": args.classifier_file,
        "blocker_file": args.blocker_file,
        "allowed_actions_source": args.allowed_actions_source,
        "blocking_mode": args.blocking_mode,
        "reward_scale": args.reward_scale,
        "catastrophe_type": args.catastrophe_type,
        
        "prior_count": args.prior_count,

        "policy": args.policy,
        "lstm_size": args.lstm_size,
        "entropy_scale": args.entropy_scale,
        "local_steps": args.local_steps,
        "death_penalty": args.death_penalty,
        "clip_norm": args.clip_norm,
        "deterministic": args.deterministic,
    }
