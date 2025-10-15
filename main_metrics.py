import argparse
from utils import load_config
from benchmarking.partition_quality_metrics import PartitionBalanceBenchmark
from graph_loader import generate_graph


def main():
    parser = argparse.ArgumentParser(description="Compare PSFs between two configurations")
    parser.add_argument("--config1", required=True, help="Path to first configuration .py file")
    parser.add_argument("--config2", required=True, help="Path to first configuration .py file")
    parser.add_argument("--subarray_number", default=2, help="Number of subarrays")
    parser.add_argument("--metric", default="l1", help="Metric to use")
    parser.add_argument("--x_param", default="node_num", help="Parameter being varied")
    args = parser.parse_args()

    config1 = load_config(args.config1)
    config2 = load_config(args.config2)

    benchmark = PartitionBalanceBenchmark(config1=config1, config2=config2,
                                          graph_func=generate_graph, output_dir="Results/", x_param=args.x_param)

    benchmark.run(subarray_num=args.subarray_number)

    benchmark.plot_metric_comparison(args.metric)


if __name__ == "__main__":
    main()
