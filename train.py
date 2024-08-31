import os
import argparse
import sys
import torch

sys.path.append("src")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("-e", "--epochs", type=int, default=70000)
    parser.add_argument("-s", "--samples", type=int, default=2500)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("-on", "--output_name", type=str, default="Schroder.pth")
    parser.add_argument("-g", "--gpuid", type=int, default=0)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)

    if args.name == "NeoHookeanExp":
        from experiments.alalytical_experiment.NeoHookeanExp import train
        train(epochs=args.epochs, n_samples=args.samples)
        print("NeoHookeanExp training finished")
    elif args.name == "Schroder":
        torch.set_default_dtype(torch.float64)
        from experiments.alalytical_experiment.SchroderNeffEbbingExp import train
        train(epochs=args.epochs, n_samples=args.samples, output_name = args.output_name)
    else:
        print("Nothing to be done.")

