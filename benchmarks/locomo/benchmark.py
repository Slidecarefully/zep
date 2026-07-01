"""Unified CLI for LOCOMO evaluation harness."""

# 这个文件是 LOCOMO benchmark 的统一命令行入口。
# 它本身不直接实现“数据如何写入 Zep 图”或“如何评测回答质量”，
# 而是负责把命令行参数、配置文件、外部客户端和具体 Runner 串起来。
#
# 整体执行主线：
#   main()
#     -> 读取 .env 环境变量
#     -> 解析命令行参数
#     -> 初始化 logger
#     -> 根据 --ingest / --eval / --cleanup 选择一个异步任务
#     -> asyncio.run(...) 执行对应流程

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# load_dotenv 会读取 .env 文件，把 ZEP_API_KEY、OPENAI_API_KEY 等变量放进 os.environ。
# 后面的 AsyncZep / AsyncOpenAI 初始化都依赖这些环境变量。
from dotenv import load_dotenv
from openai import AsyncOpenAI
from zep_cloud.client import AsyncZep

# load_config 负责把 benchmark_config.yaml 之类的配置文件解析成结构化 config。
# 三个 Runner 分别承担具体业务：
#   IngestionRunner：把 LOCOMO 数据写入 Zep graph
#   EvaluationRunner：基于 Zep graph 检索上下文并调用模型回答/评测
#   ResultsPersistence：保存每轮结果、汇总指标、实验目录
from config import load_config
from evaluation import EvaluationRunner
from ingestion import IngestionRunner
from persistence import ResultsPersistence


# 日志配置是整个 CLI 的基础设施：所有模式都会复用同一个 logger。
# 这里返回 logger，而不是直接使用 root logger，是为了让 LOCOMO benchmark 的日志有固定命名空间 "locomo"。
def setup_logging(log_level: str) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("locomo")
    logger.setLevel(getattr(logging, log_level.upper()))

    # StreamHandler 默认输出到 stderr，适合 CLI 程序实时查看运行状态。
    # formatter 统一日志格式，让 ingestion/evaluation/cleanup 的日志能用同一套时间和等级标记阅读。
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# --ingest 模式对应的执行函数。
# 它只负责“准备依赖 + 调用 IngestionRunner”，不直接处理 LOCOMO 文件细节。
async def ingest_data(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Ingest LOCOMO data into Zep using graph API."""
    # Load config
    # 配置文件决定 LOCOMO 数据范围、Zep 图参数、模型参数等。
    # ingest 阶段虽然主要写图，但仍需要统一配置来决定数据集规模和命名空间。
    config = load_config(args.config)

    # Initialize Zep client
    # Zep 是图存储/检索后端；这里使用异步客户端，因为后续 ingestion 会大量调用网络 API。
    # API key 从环境变量读取，所以 main() 必须先 load_dotenv()。
    zep = AsyncZep(api_key=os.getenv("ZEP_API_KEY"))

    # Create ingestion runner
    # prefix 用于给 graph/user 命名加命名空间，避免不同实验互相污染。
    ingestion_runner = IngestionRunner(config, zep, logger, prefix=args.prefix)

    # Ingest LOCOMO dataset
    # 真正的数据读取、session 遍历、写入 graph API 的逻辑被封装在 runner 内部。
    await ingestion_runner.ingest_locomo()

    logger.info("Ingestion complete!")


# --eval 模式对应的执行函数。
# 它比 ingest 更复杂，因为要做：展示实验配置、初始化 Zep/OpenAI、读取数据、执行一轮或多轮评测、保存结果、打印摘要。
async def evaluate_data(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Run LOCOMO evaluation using graph API."""
    # Load config
    # eval 阶段需要读取完整配置：数据规模、检索参数、回答模型、grader 模型、并发度等都会影响结果。
    config = load_config(args.config)

    # Print experimental setup
    # 这段不是业务逻辑，而是为了让每次实验在终端里自描述。
    # 后面保存结果时也会保存 config；这里先打印出来，便于人工确认当前跑的是哪套参数。
    print("\n" + "=" * 70)
    print("LOCOMO EVALUATION - EXPERIMENTAL SETUP")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Config file: {args.config}")
    print(f"\nDataset:")
    print(f"  Dataset: LOCOMO")
    print(f"  Num graphs: {config.locomo.num_users}")
    print(f"  Max sessions per graph: {config.locomo.max_session_count}")
    print(f"\nGraph Retrieval:")
    print(f"  Edge limit: {config.graph_params.edge_limit}")
    print(f"  Edge reranker: {config.graph_params.edge_reranker}")
    print(f"  Node limit: {config.graph_params.node_limit}")
    print(f"  Node reranker: {config.graph_params.node_reranker}")
    print(f"\nModels:")
    print(f"  Response model: {config.models.response_model}")
    print(f"  Response temperature: {config.models.response_temperature}")
    print(f"  Grader model: {config.models.grader_model}")
    print(f"  Grader temperature: {config.models.grader_temperature}")
    print(f"\nEvaluation:")
    print(f"  Evaluation concurrency: {config.evaluation_concurrency}")
    print(f"  Number of runs: {args.num_runs}")
    print("=" * 70 + "\n")

    # Initialize clients
    # eval 同时依赖两个外部服务：
    #   Zep：读取 graph retrieval 上下文
    #   OpenAI：生成回答或调用 grader 模型
    zep = AsyncZep(api_key=os.getenv("ZEP_API_KEY"))
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Create runners
    # EvaluationRunner 负责“对每个问题跑检索 + 回答 + 评分”的主流程。
    # ResultsPersistence 负责把结果落盘，并从原始 results 计算指标。
    evaluation_runner = EvaluationRunner(config, zep, openai_client, logger, prefix=args.prefix)
    persistence = ResultsPersistence(config, logger)

    # Load LOCOMO data
    # pandas 只在 eval 路径需要，因此放在函数内部导入，避免 ingest/cleanup 模式产生不必要依赖。
    import pandas as pd

    # 评测依赖本地 LOCOMO 数据文件。
    # 如果文件不存在，说明用户还没有准备/下载数据，或者没有先跑 ingestion 所需的数据准备流程。
    data_path = Path("data") / "locomo.json"
    if not data_path.exists():
        logger.error(f"LOCOMO data not found at {data_path}. Run ingestion first.")
        sys.exit(1)

    # LOCOMO JSON 被读成 DataFrame，后续 EvaluationRunner 可以按行遍历问题、答案、类别和元数据。
    df = pd.read_json(data_path)

    # Run evaluation multiple times
    # 这些列表是跨 run 的聚合容器：
    #   all_run_metrics：每一轮的指标对象，用于最终打印均值/方差
    #   all_run_dirs：每一轮保存目录，用于展示结果位置
    #   all_run_results：多轮实验时保存所有原始样本结果，供 experiment summary 聚合
    all_run_metrics = []
    all_run_dirs = []
    all_run_results = []  # Collect raw results from all runs

    # Create experiment directory for multi-run experiments
    # 单轮评测可以直接创建一个 timestamped run 目录；
    # 多轮评测需要先创建一个 experiment 根目录，然后每个 run 放到其子目录下，最后再保存总 summary。
    experiment_dir = None
    if args.num_runs > 1:
        experiment_dir = persistence.save_experiment(args.config)
        logger.info(f"Created experiment directory: {experiment_dir}")

    # 每一轮 run 都完整执行一次 evaluate_locomo，并单独保存结果与指标。
    # 多轮通常用于观察随机性、模型温度、检索波动或 grader 波动。
    for run_num in range(1, args.num_runs + 1):
        if args.num_runs > 1:
            print(f"\n{'=' * 70}")
            print(f"STARTING RUN {run_num}/{args.num_runs}")
            print(f"{'=' * 70}\n")
            logger.info(f"Starting evaluation run {run_num}/{args.num_runs}")

        # Run evaluation
        # evaluate_locomo 返回的是样本级结果列表；每个元素通常包含问题、模型回答、评分、检索上下文、耗时等。
        results = await evaluation_runner.evaluate_locomo(df)

        # Save results
        # 保存策略取决于是否是多轮实验：
        #   多轮：所有 run 共享 experiment_dir，run_number 用来区分子目录
        #   单轮：直接保存到一个 timestamped 目录
        if experiment_dir is not None:
            # Multi-run: save to experiment directory
            run_dir = persistence.save_run(
                results, args.config, run_number=run_num, experiment_dir=experiment_dir
            )
            # Collect raw results for aggregation
            # 多轮实验需要把每轮原始结果都累积起来，最后生成跨 run 的 experiment_summary。
            all_run_results.extend(results)
        else:
            # Single run: use timestamped directory
            run_dir = persistence.save_run(results, args.config)

        all_run_dirs.append(run_dir)

        # Calculate metrics
        # 指标计算放在保存之后，这样即使打印或聚合出错，原始结果也已经落盘。
        metrics = persistence._calculate_metrics(results)
        all_run_metrics.append(metrics)

    # Save experiment summary for multi-run experiments
    # 多轮模式下，除了每轮结果，还需要一个跨 run 的 summary，便于比较平均准确率、波动、完整上下文占比等。
    if experiment_dir is not None:
        persistence.save_experiment_summary(experiment_dir, all_run_metrics, all_run_results)

    # Print summary output
    # 终端摘要分两种：单轮打印详细指标；多轮打印聚合统计和每轮简表。
    if args.num_runs == 1:
        # Single run - print detailed metrics
        metrics = all_run_metrics[0]
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS SUMMARY")
        print("=" * 70)
        print(f"\nAccuracy: {metrics.accuracy:.3f} ({metrics.correct_count}/{metrics.total_count})")

        # Context Completeness 用来评估“检索上下文是否足以回答问题”。
        # 它和最终 accuracy 分开统计，可以帮助判断错误来自检索不足还是模型回答/推理问题。
        print("\nContext Completeness:")
        print(
            f"  COMPLETE: {metrics.completeness_complete_rate:.3f} "
            f"({metrics.completeness_complete_count}/{metrics.total_count})"
        )
        print(
            f"  PARTIAL: {metrics.completeness_partial_rate:.3f} "
            f"({metrics.completeness_partial_count}/{metrics.total_count})"
        )
        print(
            f"  INSUFFICIENT: {metrics.completeness_insufficient_rate:.3f} "
            f"({metrics.completeness_insufficient_count}/{metrics.total_count})"
        )
        # 如果存在“上下文完整时的准确率”，就单独打印。
        # 这个指标能回答：当检索已经给够信息时，生成模型本身答得怎么样。
        if metrics.accuracy_with_complete_context is not None:
            print(
                f"  Accuracy w/ Complete Context: {metrics.accuracy_with_complete_context:.3f} "
                f"({metrics.correct_with_complete_context}/{metrics.total_with_complete_context})"
            )

        # Latency Statistics 把回答耗时和检索耗时分开打印，便于定位瓶颈在 retrieval 还是 response generation。
        print("\nLatency Statistics:")
        print(
            f"  Response time - median: {metrics.response_duration_stats.median:.3f}s, "
            f"p95: {metrics.response_duration_stats.p95:.3f}s, "
            f"p99: {metrics.response_duration_stats.p99:.3f}s"
        )
        print(
            f"  Retrieval time - median: {metrics.retrieval_duration_stats.median:.3f}s, "
            f"p95: {metrics.retrieval_duration_stats.p95:.3f}s, "
            f"p99: {metrics.retrieval_duration_stats.p99:.3f}s"
        )

        # Context Token Statistics 反映检索上下文大小。
        # 它通常和准确率、延迟、成本有关：上下文太少可能信息不足，太多可能变慢或干扰模型。
        print("\nContext Token Statistics:")
        print(
            f"  Tokens - median: {metrics.context_token_stats.median:.0f}, "
            f"mean: {metrics.context_token_stats.mean:.0f}, "
            f"p95: {metrics.context_token_stats.p95:.0f}, "
            f"p99: {metrics.context_token_stats.p99:.0f}"
        )

        # 分类维度的准确率可以帮助定位某些问题类型是否特别难，或某类检索上下文是否不足。
        print("\nBy Category:")
        for cat_metrics in metrics.by_category:
            print(
                f"  Category {cat_metrics.category}: {cat_metrics.accuracy:.3f} "
                f"({cat_metrics.correct_count}/{cat_metrics.total_count})"
            )

        print(f"\nResults saved to: {all_run_dirs[0]}")
        print("=" * 70 + "\n")
    else:
        # Multiple runs - print aggregated statistics
        # 多轮聚合只在这里才需要 statistics，因此延迟导入。
        from statistics import mean, stdev

        print("\n" + "=" * 70)
        print(f"EVALUATION RESULTS SUMMARY - {args.num_runs} RUNS")
        print("=" * 70)

        # Aggregate accuracy statistics
        # 多轮准确率不只看均值，也看标准差、最小值、最大值，判断实验结果是否稳定。
        accuracies = [m.accuracy for m in all_run_metrics]
        print(f"\nAccuracy:")
        print(f"  Mean: {mean(accuracies):.3f}")
        if len(accuracies) > 1:
            print(f"  Std Dev: {stdev(accuracies):.3f}")
        print(f"  Min: {min(accuracies):.3f}")
        print(f"  Max: {max(accuracies):.3f}")
        print(f"  Runs: {[f'{a:.3f}' for a in accuracies]}")

        # Aggregate completeness statistics
        # completeness 聚合用于观察多轮下检索质量是否稳定，而不是只看最终答案正确率。
        complete_rates = [m.completeness_complete_rate for m in all_run_metrics]
        partial_rates = [m.completeness_partial_rate for m in all_run_metrics]
        insufficient_rates = [m.completeness_insufficient_rate for m in all_run_metrics]

        print(f"\nContext Completeness (Mean):")
        print(f"  COMPLETE: {mean(complete_rates):.3f}")
        print(f"  PARTIAL: {mean(partial_rates):.3f}")
        print(f"  INSUFFICIENT: {mean(insufficient_rates):.3f}")

        # Aggregate accuracy with complete context
        # 有些 run 可能没有这个派生指标，所以先过滤 None。
        complete_ctx_accuracies = [
            m.accuracy_with_complete_context
            for m in all_run_metrics
            if m.accuracy_with_complete_context is not None
        ]
        if complete_ctx_accuracies:
            print(f"\nAccuracy w/ Complete Context:")
            print(f"  Mean: {mean(complete_ctx_accuracies):.3f}")
            if len(complete_ctx_accuracies) > 1:
                print(f"  Std Dev: {stdev(complete_ctx_accuracies):.3f}")

        # Per-run details
        # 除了聚合值，也保留每轮简要结果，方便快速发现某一轮异常偏高或偏低。
        print(f"\nPer-Run Results:")
        for idx, metrics in enumerate(all_run_metrics, 1):
            print(
                f"  Run {idx}: Accuracy={metrics.accuracy:.3f}, "
                f"Complete={metrics.completeness_complete_rate:.3f}"
            )

        # Show experiment directory location
        # 多轮实验的文件结构比单轮复杂，因此明确打印根目录、summary、config 和各 run 结果文件名。
        print(f"\nExperiment directory: {experiment_dir}")
        print(f"Experiment summary: {experiment_dir / 'experiment_summary.json'}")
        print(f"Configuration: {experiment_dir / 'config.yaml'}")
        print(f"Run results: {', '.join([f'run_{i+1}_results.json' for i in range(args.num_runs)])}")

        print("\n" + "=" * 70 + "\n")

    logger.info(f"Evaluation complete. {args.num_runs} run(s) saved.")


# --cleanup 模式对应的执行函数。
# 它先列出所有 Zep graphs，再筛选当前 prefix 命名空间下的实验图；只有传入 --delete 时才会执行删除。
async def cleanup_users(args: argparse.Namespace, logger: logging.Logger) -> None:
    """List and optionally delete all graphs from Zep with the specified prefix."""
    # Initialize Zep client
    # cleanup 只需要访问 Zep，不需要 OpenAI 客户端。
    zep = AsyncZep(api_key=os.getenv("ZEP_API_KEY"))

    logger.info("Fetching all graphs...")

    # List all graphs with pagination
    # Zep graph.list 是分页接口，因此这里手动循环 page_number，直到没有更多结果。
    all_graphs = []
    page_number = 1
    page_size = 100

    while True:
        # 每次拉一页 graph；page_size 设为 100 是为了减少 API 往返，同时避免单次返回过大。
        result = await zep.graph.list(page_size=page_size, page_number=page_number)
        if not result.graphs:
            break
        all_graphs.extend(result.graphs)
        page_number += 1

        # Break if we've fetched all graphs
        # 如果当前页数量小于 page_size，说明已经到最后一页，不需要再请求下一页。
        if len(result.graphs) < page_size:
            break

    # Filter for graphs with the specified prefix
    # ingest/eval 使用 prefix 构造 graph_id，这里用同样的命名规则找到当前实验命名空间下的 graphs。
    prefix_pattern = f"{args.prefix}_experiment_graph_"
    prefix_graphs = [g for g in all_graphs if g.graph_id.startswith(prefix_pattern)]

    if not prefix_graphs:
        logger.info(f"No graphs found with prefix '{args.prefix}'.")
        return

    # 先列出将要处理的 graphs，让用户确认 cleanup 作用范围。
    logger.info(f"Found {len(prefix_graphs)} graphs with prefix '{args.prefix}':")
    for graph in prefix_graphs:
        logger.info(f"  - {graph.graph_id}")

    # Ask for confirmation if delete flag is set
    # cleanup 默认只列出，不删除；只有显式 --delete 才进入危险操作。
    if args.delete:
        logger.warning(f"About to delete {len(prefix_graphs)} graphs with prefix '{args.prefix}'.")
        confirmation = input("Type 'yes' to confirm deletion: ")
        if confirmation.lower() != "yes":
            logger.info("Deletion cancelled.")
            return

        # Delete graphs
        # 删除逐个执行，这样单个 graph 删除失败不会阻断后续 graph 的尝试。
        logger.info("Deleting graphs...")
        deleted_count = 0
        for graph in prefix_graphs:
            try:
                await zep.graph.delete(graph.graph_id)
                deleted_count += 1
                logger.debug(f"Deleted graph: {graph.graph_id}")
            except Exception as e:
                logger.error(f"Failed to delete graph {graph.graph_id}: {e}")

        logger.info(f"Successfully deleted {deleted_count}/{len(prefix_graphs)} graphs.")
    else:
        # 没有 --delete 时只提示用户如何执行真正删除。
        logger.info("Use --delete flag to delete these graphs.")


# CLI 的总入口。
# 这个函数只做命令行层面的编排：环境变量、参数解析、日志、模式分发、异常出口。
def main() -> None:
    """Main CLI entry point."""
    # Load environment variables
    # 必须在创建 AsyncZep / AsyncOpenAI 前执行，否则 os.getenv(...) 可能拿不到 API key。
    load_dotenv()

    # Create parser
    # RawDescriptionHelpFormatter 会保留 epilog 里的换行和缩进，让示例命令更易读。
    parser = argparse.ArgumentParser(
        description="LOCOMO Evaluation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest LOCOMO data using graph API
  python benchmark.py --ingest

  # Run single evaluation
  python benchmark.py --eval

  # Run multiple evaluations (creates separate experiment for each run)
  python benchmark.py --eval --num-runs 3

  # Use custom prefix for namespacing experiments
  python benchmark.py --ingest --prefix experiment_a
  python benchmark.py --eval --prefix experiment_a

  # Run multiple evaluations with custom config
  python benchmark.py --eval --num-runs 5 --config benchmark_config.yaml

  # List LOCOMO graphs (default prefix)
  python benchmark.py --cleanup

  # List graphs with custom prefix
  python benchmark.py --cleanup --prefix experiment_a

  # Delete LOCOMO graphs
  python benchmark.py --cleanup --delete

  # Delete graphs with custom prefix
  python benchmark.py --cleanup --prefix experiment_a --delete

  # Run with debug logging
  python benchmark.py --eval --log-level DEBUG
        """,
    )

    # Mode selection (mutually exclusive)
    # 三个模式互斥且必须选一个，避免用户同时触发 ingest/eval/cleanup 造成状态混乱。
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--ingest", action="store_true", help="Ingest data into Zep using graph API")
    mode_group.add_argument("--eval", action="store_true", help="Run evaluation")
    mode_group.add_argument(
        "--cleanup", action="store_true", help="List or delete LOCOMO graphs from Zep"
    )

    # Common arguments
    # 这些参数对多个模式都有意义：
    #   --config：控制实验配置
    #   --log-level：控制日志详细程度
    #   --prefix：给 Zep graph/user 命名加命名空间
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_config.yaml",
        help="Path to configuration file (default: benchmark_config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="locomo",
        help="Prefix for user/graph names to namespace experiments (default: locomo)",
    )

    # Evaluation-specific arguments
    # num-runs 只在 --eval 时真正使用；ingest/cleanup 模式解析它但不会读取。
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of evaluation runs to perform (default: 1). Each run creates a separate experiment.",
    )

    # Cleanup-specific arguments
    # --delete 是 cleanup 的安全开关：没有它时只列出 graphs，不会执行删除。
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete users when using --cleanup (requires confirmation)",
    )

    # Parse arguments
    # argparse 会在这里完成互斥模式校验、类型转换和默认值填充。
    args = parser.parse_args()

    # Setup logging
    # 日志等级来自命令行参数，之后传给各个 runner 保持统一输出。
    logger = setup_logging(args.log_level)

    # Run appropriate mode
    # 三条业务路径都是 async 函数；main 是同步入口，所以用 asyncio.run 创建事件循环并执行。
    try:
        if args.ingest:
            asyncio.run(ingest_data(args, logger))
        elif args.eval:
            asyncio.run(evaluate_data(args, logger))
        elif args.cleanup:
            asyncio.run(cleanup_users(args, logger))
    except KeyboardInterrupt:
        # Ctrl+C 属于用户主动中断，记录简洁日志后用非零退出码结束。
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        # 其他异常记录完整 traceback，便于定位 runner、API 或配置错误。
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


# Python 脚本直接运行时进入 CLI；被其他模块 import 时不会自动执行。
if __name__ == "__main__":
    main()
