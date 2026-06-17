"""Data ingestion for LOCOMO evaluation harness using graph.add API."""

# 这个模块负责把 LOCOMO 原始数据集下载到本地，并按用户/会话/消息的层级写入 Zep Graph。
# 从整体流程看，它不是单纯的数据读取脚本，而是 evaluation harness 的“数据准备阶段”：
# 先保证本地有一份原始数据快照，再把每个 LOCOMO graph 写入远端 Zep，供后续评估阶段检索使用。

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm.asyncio import tqdm
from zep_cloud.client import AsyncZep

from config import BenchmarkConfig
from constants import DATA_DIR
from ontology import ZEP_NODE_ONTOLOGY_V2


class IngestionRunner:
    """Handles data ingestion for LOCOMO dataset using graph.add API."""

    # IngestionRunner 把“配置、Zep 客户端、日志器、实验前缀”组合在一起，
    # 这样外部 CLI 只需要创建 runner 并调用 ingest_locomo()，
    # 具体的下载、建图、设置 ontology、写入消息等细节都收敛在这个类内部。
    def __init__(
        self,
        config: BenchmarkConfig,
        zep_client: AsyncZep,
        logger: logging.Logger,
        prefix: str = "locomo",
    ):
        # 保存 benchmark 配置，后续会从中读取：
        # 数据集 URL、要导入的图数量、每个图最多处理多少 session、并发度等参数。
        self.config = config

        # 复用外部已经初始化好的 AsyncZep 客户端，
        # 避免在每个 graph 或每条 message 写入时重复创建网络客户端。
        self.zep = zep_client

        # 日志器由 CLI 或上层程序传入，便于所有 ingestion 日志使用统一格式和等级。
        self.logger = logger

        # prefix 用来给 graph_id 做命名空间隔离。
        # 例如同一套 LOCOMO 数据可以用不同 prefix 导入成多组实验图，互不冲突。
        self.prefix = prefix

        # 用信号量限制同时导入 graph 的数量。
        # graph.add 是远程 API 调用，如果一次性并发太高，容易触发速率限制或网络抖动；
        # 因此这里用 config.ingestion_concurrency 控制吞吐和稳定性之间的平衡。
        self._semaphore = asyncio.Semaphore(config.ingestion_concurrency)

    async def ingest_locomo(self) -> pd.DataFrame:
        """Ingest LOCOMO dataset."""
        # ingest_locomo 是整个导入流程的公开入口：
        # 它先下载并缓存 LOCOMO 数据，再为每个用户/graph 启动一个异步导入任务。
        self.logger.info("Downloading LOCOMO dataset...")

        # Download data
        # 配置文件中维护数据集地址，runner 不硬编码 URL，
        # 这样替换数据源或切换实验配置时无需改代码。
        url = self.config.locomo.data_url

        # 先用 requests 下载原始 JSON。
        # timeout=30 避免网络卡住时无限等待；raise_for_status() 则把 HTTP 错误显式抛出。
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # data 用于原样保存到本地，保证后续可以复现当次下载的数据内容。
        data = response.json()

        # locomo_df 用于后续按 DataFrame 结构读取 conversation 字段。
        # 这里再次从 URL 读取为 DataFrame，保持后续索引访问方式简单。
        locomo_df = pd.read_json(url)

        # Save locally
        # 在写入远端 Zep 之前，先把原始数据保存到本地 data 目录。
        # 这一步的价值是：评估阶段可以直接读取本地 locomo.json，
        # 不必每次评估都重新下载远程数据，也便于排查导入和评估是否使用同一份数据。
        os.makedirs(DATA_DIR, exist_ok=True)
        data_path = Path(DATA_DIR) / "locomo.json"
        with open(data_path, "w") as f:
            json.dump(data, f, indent=2)
        self.logger.info(f"Saved dataset to {data_path}")

        # Ingest into Zep
        # 根据配置中的 num_users 决定要导入多少个 LOCOMO graph。
        # 每个 group_idx 对应 DataFrame 中的一条 conversation，
        # 也会对应 Zep 中一个独立的 graph_id。
        self.logger.info(f"Ingesting {self.config.locomo.num_users} graphs...")
        tasks = []
        for group_idx in range(self.config.locomo.num_users):
            # 这里只创建 coroutine，不立即 await。
            # 这样可以先收集所有 graph 导入任务，再交给 asyncio.as_completed 并发推进。
            task = self._ingest_locomo_graph(locomo_df, group_idx)
            tasks.append(task)

        # Process with progress bar
        # as_completed 会在任意一个任务完成时返回该任务，
        # 所以进度条反映的是“已完成 graph 数”，而不是提交任务数。
        # 每个任务内部还会通过 semaphore 限制实际并发量。
        with tqdm(total=len(tasks), desc="Ingesting graphs (v2)", unit="graph") as pbar:
            for coro in asyncio.as_completed(tasks):
                await coro
                pbar.update(1)

        # 所有 graph 写入结束后返回 DataFrame，
        # 让调用方如果需要可以继续复用已加载的数据。
        self.logger.info("LOCOMO ingestion (v2) complete")
        return locomo_df

    async def _ingest_locomo_graph(self, df: pd.DataFrame, group_idx: int) -> bool:
        """Ingest a single LOCOMO graph using graph.add API with graph_id."""
        # 这个私有方法负责导入单个 LOCOMO graph。
        # 它被 ingest_locomo() 为多个 group_idx 并发调用，
        # 但进入实际远程操作前必须先拿到 semaphore，避免同时写入过多图。
        async with self._semaphore:
            try:
                # 每一行 conversation 代表一个 LOCOMO 用户/实验图的完整多轮会话数据。
                conversation = df["conversation"].iloc[group_idx]

                # graph_id 由 prefix 和 group_idx 组成。
                # 这样同一次实验中的图可以被批量识别，也方便 cleanup 阶段按 prefix 删除。
                graph_id = f"{self.prefix}_experiment_graph_{group_idx}"

                # Create graph - ignore if exists
                # 先确保远端 graph 存在。
                # 如果 graph 已经存在，导入流程不会直接失败，而是继续使用已有 graph。
                # 这种设计适合重复运行 ingestion，但也意味着重复导入时可能追加重复 message，
                # 因此真正是否允许重复数据，需要结合 graph.add 的幂等策略或外部清理流程来看。
                try:
                    await self.zep.graph.create(
                        graph_id=graph_id,
                        name=f"LOCOMO Graph {group_idx}",
                        description=f"Multi-participant conversation graph for LOCOMO experiment {group_idx}",
                    )
                    self.logger.debug(f"Created graph: {graph_id}")
                except Exception as e:
                    self.logger.debug(f"Graph {graph_id} already exists: {e}")

                # Set ontology for this graph before adding any data
                # 在写入消息前设置 ontology，是因为后续 graph.add 会基于图的 schema/ontology
                # 抽取实体和关系。先设置 ontology 可以让导入的数据按照预期节点类型组织。
                try:
                    await self.zep.graph.set_ontology(
                        entities=ZEP_NODE_ONTOLOGY_V2,
                        edges={},
                        graph_ids=[graph_id],
                    )
                    self.logger.debug(f"Set ontology for graph: {graph_id}")
                except Exception as e:
                    # ontology 设置失败时不能继续写数据：
                    # 如果没有正确 ontology，后续导入可能产生错误结构或不一致的图。
                    self.logger.error(
                        f"Failed to set ontology for graph {graph_id}: {e}"
                    )
                    raise

                # Process each session - add messages directly to graph
                # LOCOMO conversation 按 session_0、session_1 等字段组织。
                # 这里按 session_idx 顺序处理，保留原始对话在时间上的推进关系。
                for session_idx in range(self.config.locomo.max_session_count):
                    session_key = f"session_{session_idx}"
                    session = conversation.get(session_key)

                    # 某些 conversation 可能没有达到 max_session_count；
                    # 缺失的 session 直接跳过，而不是中断整个 graph 导入。
                    if session is None:
                        continue

                    # Parse session timestamp
                    # 每个 session 有对应的 session_X_date_time 字段。
                    # LOCOMO 原始时间字符串没有直接带标准时区对象，
                    # 所以这里拼接 " UTC" 后按固定格式解析，并显式设置 UTC 时区。
                    session_date = (
                        conversation.get(f"session_{session_idx}_date_time") + " UTC"
                    )
                    date_format = "%I:%M %p on %d %B, %Y UTC"
                    date_string = datetime.strptime(session_date, date_format).replace(
                        tzinfo=UTC
                    )

                    # graph.add 需要可序列化的时间字段；
                    # isoformat() 把 datetime 转为标准 ISO 字符串，供 Zep 记录 created_at。
                    iso_date = date_string.isoformat()

                    # Process each message in the session
                    # 一个 session 内包含多条消息。
                    # 这里保留消息原有顺序逐条写入，因为顺序会影响对话上下文的构建。
                    for msg in session:
                        # speaker 表示消息是谁说的，text 是正文。
                        # blip_captions 用来补充图片描述，让图构建时也能看到图片语义。
                        speaker = msg.get("speaker")
                        text = msg.get("text")
                        blip_caption = msg.get("blip_captions")

                        # 默认以文本消息作为 graph.add 的内容主体。
                        content = text

                        # 如果消息带有图片描述，把图片语义拼接到文本后面。
                        # 这样即使 graph.add 只接收字符串，也不会丢失 multimodal 数据中的图片线索。
                        if blip_caption:
                            content += (
                                f" (description of attached image: {blip_caption})"
                            )

                        # Determine role based on speaker
                        # LOCOMO 中说话人是 "User" 时标记为 user，
                        # 其他说话人统一作为 assistant。
                        # 这里的 role 主要服务于 message_data 的结构化表达，
                        # 让后续图抽取或检索能区分用户侧和助手/其他参与者侧的发言。
                        role = "user" if speaker == "User" else "assistant"

                        # Format message as string for graph.add
                        # graph.add 写入的是一段字符串，因此这里把 speaker、role、content 合并。
                        # 这种格式既保留了原始说话人，也提供了标准化角色信息。
                        message_data = f"{speaker} ({role}): {content}"

                        # Add message to graph using graph.add API
                        # 每条消息都以 type="message" 写入同一个 graph_id。
                        # created_at 使用 session 级别时间，因此同一 session 内消息共享时间戳；
                        # 会话内部的先后关系则由写入顺序和文本内容共同体现。
                        await self.zep.graph.add(
                            graph_id=graph_id,
                            type="message",
                            data=message_data,
                            created_at=iso_date,
                        )

                # 走到这里说明当前 graph 的所有 session/message 都已尝试写入完成。
                return True

            except Exception as e:
                # 单个 graph 导入失败时只记录错误并返回 False，
                # 不把异常继续抛到外层，是为了避免一个 graph 的问题中断整个批量导入。
                # 外层当前只更新进度条，没有统计失败数；如果需要严格验收，可在这里向上返回状态并汇总。
                self.logger.error(f"Failed to ingest LOCOMO graph {group_idx}: {e}")
                return False
