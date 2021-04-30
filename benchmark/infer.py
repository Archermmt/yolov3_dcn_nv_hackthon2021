import argparse
from infer_util import *

def main(args):
    platform = args.platform
    model_dir = args.model_dir
    batch = args.batch
    run_benchmark = args.run_benchmark
    param_type = args.param_type
    warm_up = args.warm_up
    debug = args.debug

    if run_benchmark == False and batch != 1:
        batch = 1
        print("[WARNING] if run_benchmark is False, batch must be equal 1")
    model = None
    if platform == "Paddle":
        model  = infer_paddle(model_dir, batch, run_benchmark, param_type, warm_up, debug)
    elif platform == "QuakeRT":
        model = infer_quakert(model_dir, batch, run_benchmark, param_type, warm_up, debug)
    else:
        print("[ERROR] Not support platform")
        return

    model.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark describe.')
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="Paddle",
        help="Test platform type, support Paddle or QuakeRT, default is Paddle.")
    parser.add_argument(
        "-m",
        "--model_dir",
        type = str,
        default="/usr/local/quake/datas/benchmark/model",
        help="Test model path, default is /usr/local/quake/datas/benchmark/model.")
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=1,
        help="Test batch size, default is 1.")

    parser.add_argument(
        "-t",
        "--param_type",
        type=str,
        default="fp32",
        help="Inference data type, support fpr32 | fp16 | int8,default is fp32.")

    parser.add_argument(
        "-r",
        "--run_benchmark",
        type=bool,
        default=False,
        help="run benchmark test, default is False.")

    parser.add_argument(
        "-w",
        "--warm_up",
        type=int,
        default=100,
        help="Warm-up times, default is 100.")

    parser.add_argument(
        "-d",
        "--debug",
        type=bool,
        default=False,
        help="open debug, default is False.")

    args = parser.parse_args()
    print(args)
    main(args)