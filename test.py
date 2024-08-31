import argparse
import sys
import torch

sys.path.append("src")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--modelname", type=str, default="")

    args = parser.parse_args()
    if args.name == "NeoHookeanExp":
        from experiments.alalytical_experiment.NeoHookeanExp import predict
        predict()
        print("NeoHookeanExp test finished")
    elif args.name == "Schroder":
        from experiments.alalytical_experiment.SchroderNeffEbbingExp import predict

        torch.set_default_dtype(torch.float64)
        predict(modelname = args.modelname)
        print("SchroderNeffEbbingExp test finished")
    else:
        print("Nothing to be done.")

