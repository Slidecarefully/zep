"""Unified CLI for LOCOMO evaluation harness."""

# 这个文件是 LOCOMO benchmark 的统一命令行入口。
# 整体逻辑不是直接在 main 中写所有流程，而是先定义三个异步业务函数：
# ingest_data 负责把 LOCOMO 数据写入 Zep 图；evaluate_data 负责跑评估并保存结果；
# cleanup_users 负责列出或删除带指定前缀的图。
# main() 只负责解析命令行参数、初始化日志，然后根据互斥模式分发到对应业务函数。

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# .env 用来把 ZEP_API_KEY、OPENAI_API_KEY 等本地环境变量加载到 os.environ。
from dotenv import load_dotenv

# AsyncOpenAI 用于评估阶段调用 OpenAI 模型生成回答或打分。
from openai import AsyncOpenAI

# AsyncZep 是 Zep Cloud 的异步客户端，ingest/eval/cleanup 三个模式都会围绕它访问图数据。
from zep_cloud.client import AsyncZep

# 以下几个本地模块把具体业务拆开：
# config 负责读取 benchmark 配置；evaluation/ingestion 分别封装评估和导入流程；
# persistence 负责把每轮结果、实验汇总和配置快照落盘。
from config import load_config
from evaluation import EvaluationRunner
from ingestion import IngestionRunner
from persistence import ResultsPersistence


def setup_logging(log_level: str) -> logging.Logger:
    """Setup logging configuration."""
    # 统一使用名为 locomo 的 logger，让整套 CLI 的日志来源保持一致。
    # log_level 来自命令行参数，先转成大写，再映射到 logging.DEBUG/INFO 等常量。
    logger = logging.getLogger("locomo")
    logger.setLevel(getattr(logging, log_level.upper()))

    # 日志默认输出到标准错误流，适合 CLI 场景；格式中包含时间、logger 名称、级别和消息。
    # 后续业务函数只拿 logger 打日志，不需要关心输出到哪里、格式是什么。
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # 返回配置好的 logger，作为依赖传给 ingest/eval/cleanup，避免每个函数重复创建。
    return logger


async def ingest_data(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Ingest LOCOMO data into Zep using graph API."""
    # 入口参数只保存命令行层面的选择；真正控制导入行为的参数放在配置文件中。
    # 因此第一步先读取 args.config 指向的配置，再把配置交给 runner。
    # Load config
    config = load_config(args.config)

    # Zep 客户端只需要 API key。这里假设 main() 已经调用 load_dotenv，
    # 所以 os.getenv 可以同时读取系统环境变量和 .env 文件中的变量。
    # Initialize Zep client
    zep = AsyncZep(api_key=os.getenv("ZEP_API_KEY"))

    # IngestionRunner 汇总了导入 LOCOMO 的所有细节。
    # prefix 用于命名空间隔离：不同实验可以使用不同前缀，避免图 ID 互相污染。
    # Create ingestion runner
    ingestion_runner = IngestionRunner(config, zep, logger, prefix=args.prefix)

    # 真正的数据读取、图创建、节点/边写入都封装在 ingest_locomo() 里。
    # 这里 await 它，保证导入完成后才打印完成日志。
    # Ingest LOCOMO dataset
    await ingestion_runner.ingest_locomo()

    logger.info("Ingestion complete!")


async def evaluate_data(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Run LOCOMO evaluation using graph API."""
    # 评估流程首先读取配置。后面的模型选择、检索参数、并发度、数据规模都来自这里。
    # Load config
    config = load_config(args.config)

    # 先把实验配置打印到终端，是为了让每次评估的上下文在日志/控制台中可追溯。
    # 这些信息不会影响运行逻辑，但能帮助比较不同配置或多轮实验的差异。
    # Print experimental setup
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

    # 评估阶段同时需要访问 Zep 图数据和 OpenAI 模型：
    # Zep 负责检索上下文，OpenAI 负责生成回答/可能也参与 grader。
    # Initialize clients
    zep = AsyncZep(api_key=os.getenv("ZEP_API_KEY"))
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # EvaluationRunner 负责“如何评估”，ResultsPersistence 负责“如何保存”。
    # 这样运行流程可以先拿到 results，再根据单轮/多轮实验选择不同保存方式。
    # Create runners
    evaluation_runner = EvaluationRunner(config, zep, openai_client, logger, prefix=args.prefix)
    persistence = ResultsPersistence(config, logger)

    # LOCOMO 原始数据用 pandas 读取成 DataFrame。
    # import 放在函数内部可以让非评估模式不必加载 pandas，减少 CLI 其他模式的依赖成本。
    # Load LOCOMO data
    import pandas as pd

    data_path = Path("data") / "locomo.json"
    if not data_path.exists():
        # 评估依赖本地数据文件；如果缺失，直接退出并提示先运行 ingestion。
        # 这里用 sys.exit(1) 明确告诉 shell 此次命令失败。
        logger.error(f"LOCOMO data not found at {data_path}. Run ingestion first.")
        sys.exit(1)

    df = pd.read_json(data_path)

    # 这些列表贯穿整个评估循环：
    # all_run_metrics 保存每轮聚合指标，all_run_dirs 保存每轮结果目录，
    # all_run_results 只在多轮实验时用于最终汇总原始结果。
    # Run evaluation multiple times
    all_run_metrics = []
    all_run_dirs = []
    all_run_results = []  # Collect raw results from all runs

    # 单轮评估可以直接保存到一个时间戳目录；多轮评估则先创建一个实验总目录，
    # 后续每一轮都放在这个总目录下，最后再生成跨轮 summary。
    # Create experiment directory for multi-run experiments
    experiment_dir = None
    if args.num_runs > 1:
        experiment_dir = persistence.save_experiment(args.config)
        logger.info(f"Created experiment directory: {experiment_dir}")

    # run_num 从 1 开始，是为了输出和文件命名更贴近人类阅读习惯。
    # 每轮都复用同一个 evaluation_runner 和同一个 DataFrame，差异主要来自模型随机性或外部服务状态。
    for run_num in range(1, args.num_runs + 1):
        if args.num_runs > 1:
            # 多轮时打印醒目的分隔符，避免控制台中不同 run 的日志混在一起难以定位。
            print(f"\n{'=' * 70}")
            print(f"STARTING RUN {run_num}/{args.num_runs}")
            print(f"{'=' * 70}\n")
            logger.info(f"Starting evaluation run {run_num}/{args.num_runs}")

        # evaluate_locomo 是每轮评估的核心：它遍历数据集中的问题，执行检索、回答、打分等步骤，
        # 最终返回当前 run 的原始结果列表。
        # Run evaluation
        results = await evaluation_runner.evaluate_locomo(df)

        # 保存策略根据是否存在 experiment_dir 分流。
        # experiment_dir 不为 None 表示多轮实验，每轮需要带 run_number 存入同一个实验目录；
        # 否则就是单轮实验，直接创建一个独立的时间戳目录。
        # Save results
        if experiment_dir is not None:
            # Multi-run: save to experiment directory
            run_dir = persistence.save_run(
                results, args.config, run_number=run_num, experiment_dir=experiment_dir
            )
            # 多轮实验不仅要保存每轮文件，还要把每轮原始结果攒起来，供 save_experiment_summary 做总体汇总。
            # Collect raw results for aggregation
            all_run_results.extend(results)
        else:
            # Single run: use timestamped directory
            run_dir = persistence.save_run(results, args.config)

        all_run_dirs.append(run_dir)

        # 指标计算在保存之后执行，逻辑上把“结果落盘”和“指标展示/汇总”分开。
        # 这里调用的是 persistence 的内部计算函数，说明指标结构和保存逻辑被放在同一个持久化模块中维护。
        # Calculate metrics
        metrics = persistence._calculate_metrics(results)
        all_run_metrics.append(metrics)

    # 多轮实验结束后，才有足够信息生成跨轮 summary：包括每轮指标，以及合并后的原始结果。
    # Save experiment summary for multi-run experiments
    if experiment_dir is not None:
        persistence.save_experiment_summary(experiment_dir, all_run_metrics, all_run_results)

    # 输出汇总时再次分成单轮和多轮：
    # 单轮侧重详细指标，多轮侧重统计分布和实验目录结构。
    # Print summary output
    if args.num_runs == 1:
        # 单轮时只有一个 metrics，因此直接取 all_run_metrics[0] 展示完整评估面板。
        # Single run - print detailed metrics
        metrics = all_run_metrics[0]
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS SUMMARY")
        print("=" * 70)
        print(f"\nAccuracy: {metrics.accuracy:.3f} ({metrics.correct_count}/{metrics.total_count})")

        # Context Completeness 用来区分“答案是否正确”和“检索上下文是否足够”。
        # 这能帮助判断错误来自生成模型、检索模块，还是数据上下文本身不足。
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
        if metrics.accuracy_with_complete_context is not None:
            # 只有当存在完整上下文样本时，才展示“完整上下文下的准确率”。
            # 这个条件避免没有分母时打印无意义指标。
            print(
                f"  Accuracy w/ Complete Context: {metrics.accuracy_with_complete_context:.3f} "
                f"({metrics.correct_with_complete_context}/{metrics.total_with_complete_context})"
            )

        # 延迟指标分成 response 和 retrieval，分别观察生成阶段与检索阶段的性能瓶颈。
        # median 反映典型体验，p95/p99 反映长尾慢请求。
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

        # token 统计用于判断检索上下文是否过长，或是否在不同配置下导致 prompt 成本变化。
        print("\nContext Token Statistics:")
        print(
            f"  Tokens - median: {metrics.context_token_stats.median:.0f}, "
            f"mean: {metrics.context_token_stats.mean:.0f}, "
            f"p95: {metrics.context_token_stats.p95:.0f}, "
            f"p99: {metrics.context_token_stats.p99:.0f}"
        )

        # 按 category 拆分准确率，方便定位某类问题是否显著更难。
        print("\nBy Category:")
        for cat_metrics in metrics.by_category:
            print(
                f"  Category {cat_metrics.category}: {cat_metrics.accuracy:.3f} "
                f"({cat_metrics.correct_count}/{cat_metrics.total_count})"
            )

        print(f"\nResults saved to: {all_run_dirs[0]}")
        print("=" * 70 + "\n")
    else:
        # 多轮模式需要计算跨 run 的均值、标准差、最小值、最大值。
        # mean/stdev 只在这里使用，所以局部导入即可。
        # Multiple runs - print aggregated statistics
        from statistics import mean, stdev

        print("\n" + "=" * 70)
        print(f"EVALUATION RESULTS SUMMARY - {args.num_runs} RUNS")
        print("=" * 70)

        # 先聚合核心指标 accuracy。标准差只在至少两轮时有意义。
        # Aggregate accuracy statistics
        accuracies = [m.accuracy for m in all_run_metrics]
        print(f"\nAccuracy:")
        print(f"  Mean: {mean(accuracies):.3f}")
        if len(accuracies) > 1:
            print(f"  Std Dev: {stdev(accuracies):.3f}")
        print(f"  Min: {min(accuracies):.3f}")
        print(f"  Max: {max(accuracies):.3f}")
        print(f"  Runs: {[f'{a:.3f}' for a in accuracies]}")

        # 上下文完整性也按多轮求均值，帮助观察检索配置整体是否稳定。
        # Aggregate completeness statistics
        complete_rates = [m.completeness_complete_rate for m in all_run_metrics]
        partial_rates = [m.completeness_partial_rate for m in all_run_metrics]
        insufficient_rates = [m.completeness_insufficient_rate for m in all_run_metrics]

        print(f"\nContext Completeness (Mean):")
        print(f"  COMPLETE: {mean(complete_rates):.3f}")
        print(f"  PARTIAL: {mean(partial_rates):.3f}")
        print(f"  INSUFFICIENT: {mean(insufficient_rates):.3f}")

        # 某些 run 可能没有完整上下文样本，因此先过滤 None，再决定是否打印。
        # Aggregate accuracy with complete context
        complete_ctx_accuracies = [m.accuracy_with_complete_context for m in all_run_metrics if m.accuracy_with_complete_context is not None]
        if complete_ctx_accuracies:
            print(f"\nAccuracy w/ Complete Context:")
            print(f"  Mean: {mean(complete_ctx_accuracies):.3f}")
            if len(complete_ctx_accuracies) > 1:
                print(f"  Std Dev: {stdev(complete_ctx_accuracies):.3f}")

        # 除了总体统计，也保留每一轮的摘要，方便快速发现异常 run。
        # Per-run details
        print(f"\nPer-Run Results:")
        for idx, metrics in enumerate(all_run_metrics, 1):
            print(f"  Run {idx}: Accuracy={metrics.accuracy:.3f}, Complete={metrics.completeness_complete_rate:.3f}")

        # 最后打印各类输出文件的位置，让用户能直接找到 summary、配置快照和各轮结果 JSON。
        # Show experiment directory location
        print(f"\nExperiment directory: {experiment_dir}")
        print(f"Experiment summary: {experiment_dir / 'experiment_summary.json'}")
        print(f"Configuration: {experiment_dir / 'config.yaml'}")
        print(f"Run results: {', '.join([f'run_{i+1}_results.json' for i in range(args.num_runs)])}")

        print("\n" + "=" * 70 + "\n")

    logger.info(f"Evaluation complete. {args.num_runs} run(s) saved.")


async def cleanup_users(args: argparse.Namespace, logger: logging.Logger) -> None:
    """List and optionally delete all graphs from Zep with the specified prefix."""
    # cleanup 模式和 ingest/eval 一样需要 Zep 客户端，但不需要 OpenAI 客户端，
    # 因为它只操作图列表与图删除。
    # Initialize Zep client
    zep = AsyncZep(api_key=os.getenv("ZEP_API_KEY"))

    logger.info("Fetching all graphs...")

    # Zep 的 graph.list 是分页接口，所以这里逐页拉取所有图。
    # all_graphs 先保存完整列表，后面再按 prefix 过滤，避免删除逻辑和分页逻辑耦合在一起。
    # List all graphs with pagination
    all_graphs = []
    page_number = 1
    page_size = 100

    while True:
        result = await zep.graph.list(page_size=page_size, page_number=page_number)
        if not result.graphs:
            # 当前页没有图，说明已经没有更多数据，结束分页循环。
            break
        all_graphs.extend(result.graphs)
        page_number += 1

        # 如果当前页数量小于 page_size，说明这一页已经是最后一页。
        # 这比继续请求下一页更省一次网络调用。
        # Break if we've fetched all graphs
        if len(result.graphs) < page_size:
            break

    # 只处理当前 prefix 命名空间下的实验图，避免误删别的 benchmark 或用户图。
    # 命名约定是：{prefix}_experiment_graph_ 开头。
    # Filter for graphs with the specified prefix
    prefix_pattern = f"{args.prefix}_experiment_graph_"
    prefix_graphs = [g for g in all_graphs if g.graph_id.startswith(prefix_pattern)]

    if not prefix_graphs:
        # 没有匹配图时直接返回；cleanup 的“列出”与“删除”两种模式都不需要继续执行。
        logger.info(f"No graphs found with prefix '{args.prefix}'.")
        return

    # 先列出所有匹配图，无论是否删除，用户都能确认当前 prefix 会命中哪些 graph_id。
    logger.info(f"Found {len(prefix_graphs)} graphs with prefix '{args.prefix}':")
    for graph in prefix_graphs:
        logger.info(f"  - {graph.graph_id}")

    # --cleanup 默认只是列出；只有额外传入 --delete 才进入删除流程。
    # 删除前再次交互式确认，降低误操作风险。
    # Ask for confirmation if delete flag is set
    if args.delete:
        logger.warning(f"About to delete {len(prefix_graphs)} graphs with prefix '{args.prefix}'.")
        confirmation = input("Type 'yes' to confirm deletion: ")
        if confirmation.lower() != "yes":
            logger.info("Deletion cancelled.")
            return

        # 确认后逐个删除。这里没有因为单个删除失败而终止整个流程，
        # 这样可以尽量清理掉能删除的图，并在日志中记录失败项。
        # Delete graphs
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
        # 没有 --delete 时保持只读行为，并提示用户如何触发删除。
        logger.info("Use --delete flag to delete these graphs.")


def main() -> None:
    """Main CLI entry point."""
    # main 是同步函数，异步业务函数通过 asyncio.run 启动。
    # 这样命令行入口简单清晰，也让 ingest/eval/cleanup 内部可以自然使用 await。

    # 程序一开始就加载 .env，确保后续创建 Zep/OpenAI 客户端时能读到 API key。
    # Load environment variables
    load_dotenv()

    # argparse 负责把用户输入的命令行选项转换成 args。
    # RawDescriptionHelpFormatter 会保留 epilog 中多行示例的缩进和换行。
    # Create parser
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

    # 三种运行模式互斥且必须选择一种，防止用户同时传 --ingest 和 --eval 造成流程语义冲突。
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--ingest", action="store_true", help="Ingest data into Zep using graph API")
    mode_group.add_argument("--eval", action="store_true", help="Run evaluation")
    mode_group.add_argument(
        "--cleanup", action="store_true", help="List or delete LOCOMO graphs from Zep"
    )

    # 通用参数会被多个模式共享：
    # config 控制实验配置；log-level 控制日志详细程度；prefix 控制图和实验命名空间。
    # Common arguments
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

    # num-runs 只在 --eval 下真正生效；放在公共 parser 上可以简化 argparse 结构。
    # 当值大于 1 时，evaluate_data 会自动切换到多轮实验目录和聚合 summary 逻辑。
    # Evaluation-specific arguments
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of evaluation runs to perform (default: 1). Each run creates a separate experiment.",
    )

    # --delete 只与 --cleanup 搭配使用；没有它时 cleanup 是安全的只读列表操作。
    # Cleanup-specific arguments
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete users when using --cleanup (requires confirmation)",
    )

    # 到这里，CLI 的所有结构已经定义完毕；parse_args 会根据用户输入生成 args，
    # 并自动处理 --help、非法参数、缺少互斥模式等情况。
    # Parse arguments
    args = parser.parse_args()

    # 日志级别在 args 中已经确定，因此现在创建 logger，并传给后续业务函数。
    # Setup logging
    logger = setup_logging(args.log_level)

    # 根据互斥模式分发到对应异步函数。
    # 每个分支都用 asyncio.run，把 async 函数接入同步 CLI 入口。
    # Run appropriate mode
    try:
        if args.ingest:
            asyncio.run(ingest_data(args, logger))
        elif args.eval:
            asyncio.run(evaluate_data(args, logger))
        elif args.cleanup:
            asyncio.run(cleanup_users(args, logger))
    except KeyboardInterrupt:
        # Ctrl+C 中断时给出友好日志，并以非零状态码退出，方便脚本或 CI 判断失败。
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        # 其他异常统一记录堆栈信息；exc_info=True 对排查远程 API、配置、数据文件问题很关键。
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # 只有直接执行该文件时才启动 CLI；作为模块被导入时，不会立刻解析命令行或运行任务。
    main()
