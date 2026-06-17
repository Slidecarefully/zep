"""Evaluation pipeline for LOCOMO benchmark using graph_id."""

# 这个模块负责 LOCOMO benchmark 的评估阶段。
# 它默认前置条件是：ingestion 阶段已经把每个 LOCOMO conversation 写入了对应的 Zep graph。
# 评估时，本模块不会重新建图，而是围绕每个 QA 问题执行：
#   1. 从 Zep graph 中检索 nodes 和 edges；
#   2. 把检索结果整理成上下文；
#   3. 用回答模型基于上下文生成答案；
#   4. 用 grader 模型判断答案是否正确；
#   5. 额外评估“检索上下文本身是否足够回答问题”。
# 这样最后既能看到端到端 QA accuracy，也能区分“检索没找全”和“生成模型没答好”这两类问题。

import asyncio
import logging
from time import time
from typing import Any

import pandas as pd
import tiktoken
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.asyncio import tqdm
from zep_cloud import EntityEdge, EntityNode
from zep_cloud.client import AsyncZep
from zep_cloud.core.api_error import ApiError

from common import CompletenessGrade, EvaluationResult, Grade
from config import BenchmarkConfig
from prompts import (
    CONTEXT_TEMPLATE,
    GRADER_PROMPT,
    GRADER_SYSTEM_PROMPT,
    RESPONSE_PROMPT,
    RESPONSE_SYSTEM_PROMPT,
)


class EvaluationRunner:
    """Handles evaluation for LOCOMO dataset using graph_id."""

    # EvaluationRunner 把一次评估所需的配置、Zep Cloud 客户端、OpenAI 客户端和 logger 绑定在一起。
    # 它和 ingestion runner 的职责相反：
    # ingestion 负责把原始对话写入 graph；
    # evaluation 负责读取本地 QA 数据，并用 graph.search 的结果去回答问题、打分。
    def __init__(
        self,
        config: BenchmarkConfig,
        zep_client: AsyncZep,
        openai_client: AsyncOpenAI,
        logger: logging.Logger,
        prefix: str = "locomo",
    ):
        # 保存 benchmark 配置。
        # 后续会从这里读取：评估并发数、检索 limit、reranker、回答模型、评分模型、温度等参数。
        self.config = config

        # Zep Cloud 异步客户端用于 graph.search。
        # 注意这里不直接调用 Graphiti 本地 add_episode/search，
        # 而是通过 Zep Cloud SDK 查询 ingestion 阶段已经创建好的 graph_id。
        self.zep = zep_client

        # OpenAI 异步客户端用于两类 LLM 调用：
        # 一类生成回答，一类作为 grader/完整性评估器。
        self.openai = openai_client

        # logger 由外层 CLI 传入，保持整个 benchmark harness 的日志风格一致。
        self.logger = logger

        # prefix 必须和 ingestion 阶段一致。
        # 因为 graph_id 是通过 prefix + group_idx 拼出来的；
        # 如果 prefix 不一致，evaluation 会搜索不到之前写入的数据。
        self.prefix = prefix

        # 控制同时评估多少个 QA case。
        # 单个 case 内部还会并发做 node/edge 检索和 LLM 调用；
        # 所以这里需要用 semaphore 控制总体压力，避免 Zep/OpenAI 请求过载。
        self._semaphore = asyncio.Semaphore(config.evaluation_concurrency)

        # Token counter
        # 初始化 token 计数器，用来统计拼接后的 context token 数。
        # 这个指标不直接影响答案生成，但可以帮助分析检索上下文是否过长、是否浪费 token。
        try:
            self.tokenizer = tiktoken.encoding_for_model(config.models.response_model)
        except KeyError:
            # 如果 tiktoken 不认识当前 response_model，就退回到通用 cl100k_base 编码。
            # 这样新模型名不会导致整个评估流程失败。
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    @retry(
        retry=retry_if_exception_type(ApiError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=10),
        reraise=True,
    )
    async def _graph_search_with_retry(
        self, query: str, graph_id: str, scope: str, reranker: str, limit: int
    ):
        """Wrapper for graph.search with retry logic for 503 errors."""
        # Zep graph.search 是评估链路里最关键的外部依赖之一。
        # 网络抖动或服务端临时不可用会表现为 ApiError，尤其是 503。
        # 这里用 tenacity 包一层重试，避免个别瞬时错误让整个 benchmark case 失败。
        return await self.zep.graph.search(
            query=query,
            graph_id=graph_id,
            scope=scope,
            reranker=reranker,
            limit=limit,
        )

    async def evaluate_locomo(self, df: pd.DataFrame) -> list[EvaluationResult]:
        """Evaluate LOCOMO dataset."""
        # evaluate_locomo 是评估阶段的公开入口。
        # 它负责把 DataFrame 中的所有 LOCOMO QA case 展开成异步任务，
        # 然后收集每个 case 的 EvaluationResult。
        self.logger.info(f"Evaluating {self.config.locomo.num_users} graphs...")

        all_results = []
        tasks = []

        # LOCOMO 的每个 group_idx 对应 ingestion 阶段创建的一个 graph。
        # 这里按同样规则拼 graph_id，确保问题会在对应用户/会话图里检索。
        for group_idx in range(self.config.locomo.num_users):
            qa_set = df["qa"].iloc[group_idx]
            graph_id = f"{self.prefix}_experiment_graph_{group_idx}"

            # 每个 graph 下有一组 QA。
            # 每个 QA 都会独立执行检索、回答、评分，因此可以并发评估。
            for qa_idx, qa in enumerate(qa_set):
                # Skip category 5 as golds are not provided for this category
                # LOCOMO 中 category 5 没有 gold answer，无法做准确率评分；
                # 因此这里直接跳过，避免 grader 在没有标准答案的情况下产生无意义结果。
                if qa.get("category") == 5:
                    continue

                # test_id 把 prefix、graph_idx、qa_idx 编进去，
                # 便于后续保存结果、排查单个样本、跨多次 run 对齐同一道题。
                test_id = f"{self.prefix}_graph_{group_idx}_qa_{qa_idx}"

                # 这里只创建 coroutine，不立即 await。
                # 之后用 asyncio.as_completed 逐个接收完成的任务，以便进度条实时更新。
                task = self._evaluate_locomo_conversation(graph_id, test_id, qa, qa_idx)
                tasks.append(task)

        # Process with progress bar
        # correct_count 在任务完成时动态累计，
        # 进度条 postfix 会实时显示当前已完成样本上的临时 accuracy。
        correct_count = 0
        with tqdm(total=len(tasks), desc="Evaluating", unit="test", position=0) as pbar:
            for coro in asyncio.as_completed(tasks):
                # as_completed 的好处是快完成的 case 不需要等待前面的慢 case。
                # 这对外部 API 调用很多的 benchmark 很重要，可以更平滑地推进进度。
                result = await coro
                all_results.append(result)

                # EvaluationResult.grade 是布尔值：
                # True 表示 grader 判定 hypothesis 与 gold answer 匹配。
                if result.grade:
                    correct_count += 1

                # Update metrics in progress bar
                # 这里的 accuracy 是运行中的临时统计，分母是已完成样本数。
                current_accuracy = correct_count / len(all_results)
                pbar.set_postfix(
                    {"accuracy": f"{current_accuracy:.3f}", "correct": correct_count}
                )
                pbar.update(1)

        # 所有 QA case 完成后，记录最终 accuracy。
        # all_results 会交给 persistence 层保存和进一步汇总。
        self.logger.info(
            f"Evaluation complete. Accuracy: {correct_count / len(all_results):.3f}"
        )
        return all_results

    async def _evaluate_locomo_conversation(
        self, graph_id: str, test_id: str, qa: dict[str, Any], qa_idx: int
    ) -> EvaluationResult:
        """Evaluate a single LOCOMO test case using graph_id."""
        # 单个 QA case 的完整链路都在这里：
        # 读取问题和 gold answer → 检索 graph → 组装 context → 生成 hypothesis →
        # 评估 context 完整性 → 对 hypothesis 打分 → 封装 EvaluationResult。
        async with self._semaphore:
            # 从 QA 字典中抽取评估所需字段。
            # category/difficulty 会作为分析维度保存下来，方便后续按题型或难度切片统计。
            query = qa.get("question")
            gold_answer = qa.get("answer")
            category = qa.get("category", "unknown")
            difficulty = qa.get("difficulty", "unknown")

            # Retrieval with retry logic
            # 检索阶段是 answer generation 的上游。
            # 这里分别从 nodes 和 edges 两个 scope 检索：
            # nodes 提供实体摘要，edges 提供事实/关系和事件时间。
            start_retrieval = time()
            search_results = await asyncio.gather(
                self._graph_search_with_retry(
                    query=query,
                    graph_id=graph_id,
                    scope="nodes",
                    reranker=self.config.graph_params.node_reranker,
                    limit=self.config.graph_params.node_limit,
                ),
                self._graph_search_with_retry(
                    query=query,
                    graph_id=graph_id,
                    scope="edges",
                    reranker=self.config.graph_params.edge_reranker,
                    limit=self.config.graph_params.edge_limit,
                ),
            )
            retrieval_duration = time() - start_retrieval

            # Zep graph.search 返回值按 gather 顺序排列：
            # 第一个结果是 node search，第二个结果是 edge search。
            nodes = search_results[0].nodes
            edges = search_results[1].edges

            # Compose context
            # 检索结果不能直接喂给回答模型，需要先整理成 prompt-friendly 的文本上下文。
            # 同时记录 token 和字符数，用于分析上下文规模与效果/延迟之间的关系。
            context = self._compose_context(edges, nodes)
            context_tokens = self._count_tokens(context)
            context_chars = len(context)

            # Response generation and completeness evaluation in parallel
            # 生成答案和评估 context 完整性都只依赖 query/gold/context，
            # 两者互不依赖，因此可以并行执行，减少单个样本耗时。
            # 注意完整性评估不是在评答案，而是在判断“检索到的 context 是否足够”。
            start_response = time()
            hypothesis_task = self._generate_response(context, query)
            completeness_task = self.evaluate_context_completeness(
                query, str(gold_answer), context
            )

            hypothesis, (
                completeness_grade,
                completeness_reasoning,
                missing_elements,
                present_elements,
            ) = await asyncio.gather(hypothesis_task, completeness_task)
            response_duration = time() - start_response

            # Grading
            # 等 hypothesis 生成之后，再用 grader 模型和 gold answer 对比。
            # 这一步输出的是端到端 QA 是否答对，以及判分理由。
            grade, reasoning = await self._grade_response(
                query, str(gold_answer), hypothesis
            )

            # total_duration 这里只统计 retrieval + response/completeness 并行阶段。
            # grader 调用发生在 response_duration 之后，但没有纳入 total_duration。
            # 如果要衡量完整 wall-clock latency，可以考虑把 grading 时间也单独记录。
            total_duration = retrieval_duration + response_duration

            # 将一个 QA case 的所有原始输入、检索上下文、模型输出、评分和耗时指标
            # 封装到 EvaluationResult，交由上层统一保存与汇总。
            return EvaluationResult(
                graph_id=graph_id,
                test_id=test_id,
                category=str(category),
                difficulty=str(difficulty),
                query=query,
                golden_answer=str(gold_answer),
                hypothesis=hypothesis,
                context=context,
                context_tokens=context_tokens,
                context_chars=context_chars,
                retrieval_duration=retrieval_duration,
                response_duration=response_duration,
                total_duration=total_duration,
                grade=grade,
                grade_reasoning=reasoning,
                completeness_grade=completeness_grade,
                completeness_reasoning=completeness_reasoning,
                missing_elements=missing_elements,
                present_elements=present_elements,
            )

    def _compose_context(self, edges: list[EntityEdge], nodes: list[EntityNode]) -> str:
        """Compose context from retrieved facts and entities."""
        # edges 通常是回答问题最直接的事实来源：
        # edge.fact 给出事实内容，edge.valid_at 给出事实发生或生效时间。
        # LOCOMO 问题经常有时间维度，因此 event_time 被显式放入上下文。
        facts = [f"  - {edge.fact} (event_time: {edge.valid_at})" for edge in edges]

        # nodes 则提供实体层面的背景摘要。
        # 它们未必直接回答问题，但可以补充人物、地点、对象等实体的上下文。
        entities = [f"  - {node.name}: {node.summary}" for node in nodes]

        # 最后交给统一的 CONTEXT_TEMPLATE 格式化。
        # 这样 response prompt 可以稳定地引用 facts 和 entities 两个区域。
        return CONTEXT_TEMPLATE.format(
            facts="\n".join(facts), entities="\n".join(entities)
        )

    async def _generate_response(self, context: str, question: str) -> str:
        """Generate response using LLM."""
        # 把检索上下文和问题填入回答 prompt。
        # 回答模型只看到 context + question，而不是 gold answer；
        # 因此它模拟真实 RAG/graph retrieval 问答场景。
        prompt = RESPONSE_PROMPT.format(context=context, question=question)

        # 调用 response_model 生成 hypothesis。
        # system prompt 定义回答规范，user prompt 提供具体问题和上下文。
        response = await self.openai.chat.completions.create(
            model=self.config.models.response_model,
            messages=[
                {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.models.response_temperature,
        )

        # 如果模型返回空 content，则兜底为空字符串，避免后续 grader 因 None 出错。
        return response.choices[0].message.content or ""

    async def _grade_response(
        self, question: str, gold_answer: str, response: str
    ) -> tuple[bool, str]:
        """Grade response using LLM."""
        # grader prompt 同时包含问题、gold answer 和模型回答。
        # grader 的任务不是重新回答问题，而是判断 response 是否等价满足 gold answer。
        grader_prompt = GRADER_PROMPT.format(
            question=question, gold_answer=gold_answer, response=response
        )

        # 使用 beta parse 接口，并指定 response_format=Grade。
        # 这让 grader 输出被解析成结构化对象，而不是靠字符串解析猜字段。
        grader_response = await self.openai.beta.chat.completions.parse(
            model=self.config.models.grader_model,
            messages=[
                {"role": "system", "content": GRADER_SYSTEM_PROMPT},
                {"role": "user", "content": grader_prompt},
            ],
            response_format=Grade,
            temperature=self.config.models.grader_temperature,
        )

        # Grade 里用 is_correct 字段承载分类结果。
        # 这里约定只有字符串等于 "correct" 才算 True，其余情况都按错误处理。
        result = grader_response.choices[0].message.parsed
        is_correct = result.is_correct.strip().lower() == "correct"
        return is_correct, result.reasoning

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        # token 计数用于记录 context 规模。
        # 如果 tokenizer 因异常失败，不让它影响主评估流程，只记 warning 并返回 0。
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            self.logger.warning(f"Token counting failed: {e}")
            return 0

    async def evaluate_context_completeness(
        self, question: str, gold_answer: str, context: str
    ) -> tuple[str, str, list[str], list[str]]:
        """
        Evaluate whether the retrieved context contains adequate information to answer the question.
        This is the PRIMARY evaluation metric - assessing context quality independent of the AI's answer.

        Args:
            question: The original question
            gold_answer: The expected answer (used to determine what info is needed)
            context: Retrieved context from Zep graph search

        Returns:
            Tuple of (completeness_grade, reasoning, missing_elements, present_elements)
            where completeness_grade is one of: COMPLETE, PARTIAL, INSUFFICIENT
        """
        # 这个函数评估的是“检索结果质量”，不是“回答质量”。
        # 它把 gold answer 作为判断信息需求的参照，
        # 要求 grader 检查 context 中是否包含构造 gold answer 所需的关键元素。
        instructions = """You are an expert evaluator assessing whether retrieved context contains adequate information to answer a question."""

        # input_text 是完整的 context completeness 评估任务说明。
        # 这里特别强调：
        #   - 不要评估 hypothesis；
        #   - 只看 context 是否足够；
        #   - 历史日期范围不是过期信息；
        #   - 需要区分 present_elements 和 missing_elements。
        # 这些约束是为了避免 grader 把 temporal graph 中的过去事实误判为无效。
        input_text = f"""Your task is to evaluate whether the provided CONTEXT contains sufficient information to answer the QUESTION according to what the GOLDEN ANSWER requires.

IMPORTANT: You are NOT evaluating an answer. You are evaluating whether the CONTEXT itself has the necessary information.

<QUESTION>
{question}
</QUESTION>

<GOLDEN ANSWER>
{gold_answer}
</GOLDEN ANSWER>

<CONTEXT>
{context}
</CONTEXT>

Evaluation Guidelines:

1. **COMPLETE**: The context contains ALL information needed to fully answer the question according to the golden answer.
   - All key elements from the golden answer are present
   - Sufficient detail exists to construct a complete answer
   - Historical facts (with past date ranges) ARE valid context

2. **PARTIAL**: The context contains SOME relevant information but is missing key details.
   - Some elements from the golden answer are present
   - Some critical information is missing or incomplete
   - Additional context would be needed for a complete answer

3. **INSUFFICIENT**: The context lacks most or all critical information needed.
   - Key elements from the golden answer are absent
   - Context is off-topic or irrelevant
   - No reasonable answer could be constructed from this context

IMPORTANT temporal interpretation:
- Facts with date ranges (e.g., "2025-10-01 - 2025-10-07") represent WHEN events occurred
- These historical facts remain VALID context even if dated in the past
- Only mark information as missing if it is truly ABSENT from the context
- Do NOT mark facts as "expired" or "outdated" simply because they have past dates
- Date ranges ending before "present" indicate completed/past events, not invalid information

For your evaluation:
- Identify which information elements ARE present in the context (present_elements)
- Identify which information elements are MISSING (truly absent) from the context (missing_elements)
- Historical facts (past date ranges) count as present information
- Provide clear reasoning explaining your completeness assessment

Please evaluate the context completeness:
"""

        # 同样使用结构化 parse，让输出符合 CompletenessGrade schema。
        # 这样 persistence/metrics 层可以直接读取 completeness、reasoning、
        # missing_elements、present_elements，而不用解析自由文本。
        result = await self.openai.beta.chat.completions.parse(
            model=self.config.models.grader_model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": input_text},
            ],
            response_format=CompletenessGrade,
            temperature=self.config.models.grader_temperature,
        )

        # completeness 统一转成大写，便于后续聚合统计 COMPLETE/PARTIAL/INSUFFICIENT。
        completeness_grade = (
            result.choices[0].message.parsed.completeness.strip().upper()
        )

        # 返回完整性等级、解释、缺失元素和已出现元素。
        # 这些字段让最终报告不只给出准确率，还能解释检索失败具体缺了什么。
        return (
            completeness_grade,
            result.choices[0].message.parsed.reasoning,
            result.choices[0].message.parsed.missing_elements,
            result.choices[0].message.parsed.present_elements,
        )
