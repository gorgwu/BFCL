"""Run BFCL cases through a local JiuwenSwarm agent runtime.

This adapter intentionally uses the normal JiuwenSwarm session setup and its
ProgressiveToolRail. BFCL remains responsible for the dataset and final score.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
)
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.utils import add_memory_instruction_system_prompt
from bfcl_eval.utils import contain_multi_turn_interaction, is_memory


def _workspace_paths() -> tuple[Path, Path]:
    workspace = Path(__file__).resolve().parents[5]
    swarm = Path(os.getenv("BFCL_JIUWENSWARM_ROOT", workspace / "jiuwenswarm"))
    core = Path(os.getenv("BFCL_AGENT_CORE_ROOT", workspace / "agent-core"))
    return swarm, core


def _load_jiuwenswarm():
    swarm_root, core_root = _workspace_paths()
    for root in (core_root, swarm_root):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    from jiuwenswarm.server.runtime.agent_adapter.interface import JiuWenSwarm
    from jiuwenswarm.common.config import get_config, get_progressive_tool_enabled
    from jiuwenswarm.common.utils import get_env_file
    from jiuwenswarm.dotenv_early import load_dotenv_runtime
    from jiuwenswarm.common.schema.agent import AgentRequest
    from jiuwenswarm.common.schema.message import ReqMethod
    from openjiuwen.core.foundation.tool import LocalFunction, ToolCard, ToolExposure
    from openjiuwen.core.foundation.tool.schema import ToolOutput
    from openjiuwen.core.single_agent.prompts.builder import PromptSection
    from openjiuwen.harness.rails.progressive_tool_rail import ProgressiveToolRail

    return {
        "JiuWenSwarm": JiuWenSwarm,
        "get_config": get_config,
        "get_progressive_tool_enabled": get_progressive_tool_enabled,
        "get_env_file": get_env_file,
        "load_dotenv_runtime": load_dotenv_runtime,
        "AgentRequest": AgentRequest,
        "ReqMethod": ReqMethod,
        "LocalFunction": LocalFunction,
        "ToolCard": ToolCard,
        "ToolExposure": ToolExposure,
        "ToolOutput": ToolOutput,
        "PromptSection": PromptSection,
        "ProgressiveToolRail": ProgressiveToolRail,
    }


def _normalize_schema(parameters: Any) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}}
    schema = json.loads(json.dumps(parameters))
    if schema.get("type") == "dict":
        schema["type"] = "object"
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


def _python_call(name: str, arguments: dict[str, Any]) -> str:
    encoded = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    return f"{name}({encoded})"


def _message_text(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = str(message.get("role") or "user")
        if role == "system":
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text") or "") if isinstance(part, dict) else str(part)
                for part in content
            )
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def _usage_record(raw: Any, default_model: str = "") -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    model = str(raw.get("model_name") or raw.get("model") or default_model or "")

    def first_number(*names: str) -> int | None:
        for name in names:
            value = raw.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
        return None

    input_tokens = first_number("input_tokens", "prompt_tokens")
    output_tokens = first_number("output_tokens", "completion_tokens")
    total_tokens = first_number("total_tokens")
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    cost = raw.get("cost_usd", raw.get("cost"))
    if cost is None:
        reported_total_cost = raw.get("total_cost")
        input_cost = raw.get("input_cost")
        output_cost = raw.get("output_cost")
        if any(
            isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
            for value in (reported_total_cost, input_cost, output_cost)
        ):
            cost = reported_total_cost
            if cost is None:
                cost = (input_cost or 0) + (output_cost or 0)
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        cost = None
    else:
        cost = float(cost)
    return {
        "model": model,
        "request_id": raw.get("request_id") or raw.get("id"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost,
        "usage_reported": any(
            value is not None for value in (input_tokens, output_tokens, cost)
        ),
    }


def _empty_component_usage() -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": None,
        "request_count": 0,
        "cost_reported_request_count": 0,
        "cost_complete": False,
        "models": [],
    }


def _sum_component_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = _empty_component_usage()
    total_cost = 0.0
    for record in records:
        result["request_count"] += 1
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            result[field] += int(record.get(field) or 0)
        cost = record.get("cost_usd")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            total_cost += float(cost)
            result["cost_reported_request_count"] += 1
        model = str(record.get("model") or "")
        if model and model not in result["models"]:
            result["models"].append(model)
    if result["cost_reported_request_count"]:
        result["cost_usd"] = total_cost
    result["cost_complete"] = (
        result["request_count"] > 0
        and result["cost_reported_request_count"] == result["request_count"]
    )
    return result


def _total_component_usage(*components: dict[str, Any]) -> dict[str, Any]:
    result = _empty_component_usage()
    total_cost = 0.0
    for component in components:
        result["request_count"] += component["request_count"]
        result["cost_reported_request_count"] += component[
            "cost_reported_request_count"
        ]
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            result[field] += component[field]
        if component["cost_usd"] is not None:
            total_cost += component["cost_usd"]
        for model in component["models"]:
            if model not in result["models"]:
                result["models"].append(model)
    if result["cost_reported_request_count"]:
        result["cost_usd"] = total_cost
    result["cost_complete"] = (
        result["request_count"] > 0
        and result["cost_reported_request_count"] == result["request_count"]
    )
    return result


class JiuwenSwarmHandler(BaseHandler):
    """BFCL-compatible handler that invokes a fresh JiuwenSwarm session per case."""

    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self._runtime_symbols = None
        self._loop = None
        self._loop_thread = None
        self._loop_lock = threading.Lock()

    def _symbols(self):
        if self._runtime_symbols is None:
            self._runtime_symbols = _load_jiuwenswarm()
        return self._runtime_symbols

    def inference(self, test_entry: dict, include_input_log: bool, exclude_state_log: bool):
        # Keep one event loop for this handler. BFCL invokes inference on worker
        # threads, while JiuwenSwarm and its async clients are loop-affine.
        loop = self._get_loop()
        future = asyncio.run_coroutine_threadsafe(
            self._run_case(test_entry, include_input_log), loop
        )
        return future.result()

    def _get_loop(self):
        with self._loop_lock:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                loop = self._loop

                def run_loop():
                    asyncio.set_event_loop(loop)
                    loop.run_forever()

                self._loop_thread = threading.Thread(
                    target=run_loop,
                    name="bfcl-jiuwenswarm-loop",
                    daemon=True,
                )
                self._loop_thread.start()
            return self._loop

    async def _run_case(self, test_entry: dict, include_input_log: bool):
        symbols = self._symbols()
        symbols["load_dotenv_runtime"](
            dotenv_path=symbols["get_env_file"](),
            override=True,
        )
        get_config = symbols["get_config"]
        config = get_config()
        if not symbols["get_progressive_tool_enabled"](config):
            raise RuntimeError(
                "JiuwenSwarm BFCL requires progressive_tool_enabled=true."
            )
        discovery = config.get("tool_discovery_backend", "bm25")
        discovery = str(discovery).strip().lower()
        if discovery not in {"bm25", "jev", "local"}:
            raise RuntimeError(
                "JiuwenSwarm BFCL supports tool_discovery_backend=bm25, jev, "
                "or local."
            )

        case_id = str(test_entry["id"])
        session_id = f"bfcl_{uuid.uuid4().hex}"
        model_key = re.sub(r"[-./:]", "_", self.registry_name)
        multi_turn = contain_multi_turn_interaction(case_id)
        category = case_id.rsplit("_", 1)[0]
        long_context = "long_context" in category or "composite" in category
        involved_classes = test_entry.get("involved_classes", [])
        initial_config = test_entry.get("initial_config", {})

        swarm = symbols["JiuWenSwarm"]()
        root_adapter = None
        session_adapter = None
        calls_by_turn: list[list[dict[str, Any]]] = []
        outputs_by_turn: list[str] = []
        token_input: list[list[int]] = []
        token_output: list[list[int]] = []
        latencies: list[list[float]] = []
        inference_log: list[dict[str, Any]] = []
        selected_by_turn: list[list[str]] = []
        fallback_by_turn: list[bool] = []
        current_jev_usage: list[dict[str, Any]] = []
        current_laya_usage: list[dict[str, Any]] = []
        model_usage_by_turn: list[dict[str, Any]] = []
        current_calls: list[dict[str, Any]] = []
        current_tokens = {"input": 0, "output": 0}
        benchmark_functions = list(test_entry.get("function", []))
        rail = None

        def make_tool(function: dict[str, Any]):
            name = str(function["name"])
            description = str(function.get("description") or name)
            parameters = _normalize_schema(function.get("parameters"))
            safe_id = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
            card = symbols["ToolCard"](
                id=f"bfcl_{session_id}_{safe_id}",
                name=name,
                description=description,
                input_params=parameters,
                exposure=symbols["ToolExposure"].DEFERRED,
                parallel_safe=False,
                stateless=not multi_turn,
                idempotent=False,
            )

            def execute_sync(**kwargs):
                normalized = dict(kwargs)
                current_calls.append({name: normalized})
                result_text = "BFCL tool execution is disabled for this category."
                if multi_turn:
                    execution_results, _ = execute_multi_turn_func_call(
                        [_python_call(name, normalized)],
                        initial_config,
                        involved_classes,
                        model_key,
                        case_id,
                        long_context=long_context,
                        is_evaL_run=False,
                    )
                    result_text = execution_results[0] if execution_results else ""
                return symbols["ToolOutput"](success=True, data=result_text)

            async def execute(**kwargs):
                return await asyncio.to_thread(execute_sync, **kwargs)

            return symbols["LocalFunction"](card=card, func=execute)

        def install_inventory(functions: list[dict[str, Any]]):
            nonlocal rail
            ability_manager = session_adapter._instance.ability_manager
            allowed = {
                str(function["name"])
                for function in functions
                if function.get("name")
            } | {"tool_search", "tool_call"}
            extras = [
                card.name
                for card in (ability_manager.list() or [])
                if isinstance(card, symbols["ToolCard"])
                and card.name not in allowed
            ]
            if extras:
                ability_manager.remove_ability(extras)
            for function in functions:
                tool = make_tool(function)
                ability_manager.add_ability(tool.card, tool)
            rails = [
                item
                for item in getattr(session_adapter._instance, "_registered_rails", [])
                if isinstance(item, symbols["ProgressiveToolRail"])
            ]
            if len(rails) != 1:
                raise RuntimeError(
                    f"Expected one ProgressiveToolRail, found {len(rails)}."
                )
            rail = rails[0]
            if discovery == "local":
                rail._usage_callback = lambda usage: current_laya_usage.append(dict(usage))
            rail._rebuild_tool_search_index(session_adapter._instance)

            # Fail closed if runtime setup reintroduced unrelated tools or if
            # any benchmark function/wrapper is missing or has the wrong exposure.
            cards = [
                card
                for card in (ability_manager.list() or [])
                if isinstance(card, symbols["ToolCard"])
                and getattr(card, "name", None)
            ]
            registered = {str(card.name): card for card in cards}
            expected_functions = allowed - {"tool_search", "tool_call"}
            unexpected = sorted(set(registered) - allowed)
            missing_functions = sorted(expected_functions - set(registered))
            missing_wrappers = sorted(
                {"tool_search", "tool_call"} - set(registered)
            )
            wrong_exposure = []
            for name in expected_functions & set(registered):
                exposure = getattr(registered[name], "exposure", None)
                exposure = getattr(exposure, "value", exposure)
                if str(exposure).lower() != "deferred":
                    wrong_exposure.append(name)
            for name in {"tool_search", "tool_call"} & set(registered):
                exposure = getattr(registered[name], "exposure", None)
                exposure = getattr(exposure, "value", exposure)
                if str(exposure).lower() != "direct":
                    wrong_exposure.append(name)
            if unexpected or missing_functions or missing_wrappers or wrong_exposure:
                raise RuntimeError(
                    "BFCL tool inventory check failed: "
                    f"unexpected={unexpected}, missing_functions={missing_functions}, "
                    f"missing_wrappers={missing_wrappers}, "
                    f"wrong_exposure={sorted(wrong_exposure)}"
                )

        try:
            root_adapter = swarm._ensure_adapter(mode="agent")
            root_adapter._channel_id = "web"
            await swarm.prepare_session(
                session_id=session_id,
                channel_id="web",
                mode="agent",
            )
            session_adapter = root_adapter._get_cached_session_adapter(session_id)
            if session_adapter is None or session_adapter._instance is None:
                raise RuntimeError("JiuwenSwarm did not initialize the BFCL agent session")

            if is_memory(category):
                _, instances = execute_multi_turn_func_call(
                    [], initial_config, involved_classes, model_key, case_id,
                    long_context=long_context, is_evaL_run=False,
                )
                if len(instances) != 1:
                    raise RuntimeError("BFCL memory case must initialize exactly one backend")
                test_entry["question"] = add_memory_instruction_system_prompt(
                    test_entry["question"],
                    category,
                    test_entry["scenario"],
                    next(iter(instances.values())),
                )

            install_inventory(benchmark_functions)
            original_update_session_tools = session_adapter._update_session_tools

            async def update_session_tools_with_benchmark_inventory(*args, **kwargs):
                result = await original_update_session_tools(*args, **kwargs)
                install_inventory(benchmark_functions)
                return result

            session_adapter._update_session_tools = update_session_tools_with_benchmark_inventory
            prompt_builder = getattr(
                session_adapter._instance, "system_prompt_builder", None
            )
            if prompt_builder is None:
                raise RuntimeError("JiuwenSwarm agent has no system prompt builder")
            first_turn = (test_entry.get("question") or [[]])[0]
            system_instructions = [
                str(message.get("content") or "")
                for message in first_turn
                if str(message.get("role") or "").lower() == "system"
                and message.get("content")
            ]
            if system_instructions:
                system_text = "\n\n".join(system_instructions)
                prompt_builder.add_section(
                    symbols["PromptSection"](
                        name="bfcl_case_system_instructions",
                        content={
                            "en": "BFCL case instructions:\n" + system_text,
                            "cn": "BFCL 测试用例指令：\n" + system_text,
                        },
                        priority=899,
                    )
                )
            prompt_builder.add_section(
                symbols["PromptSection"](
                    name="bfcl_benchmark_instructions",
                    content={
                        "en": (
                            "Complete the user's request using the available benchmark "
                            "functions. Call the functions that are needed, preserve "
                            "their returned results, and do not claim completion before "
                            "the requested actions have been carried out."
                        ),
                        "cn": (
                            "请使用可用的基准函数完成用户请求。按需调用函数并保留其返回结果；"
                            "在请求的操作完成前，不要声称任务已完成。"
                        ),
                    },
                    priority=900,
                )
            )

            capture_selection = rail._route_deferred_tools
            async def capture_routes(ctx):
                selected, fallback = await capture_selection(ctx)
                selected_by_turn.append([str(item.name) for item in selected])
                fallback_by_turn.append(bool(fallback))
                session = getattr(ctx, "session", None)
                discovery_state = (
                    session.get_state("__progressive_tool_discovery__")
                    if session is not None
                    else None
                )
                usage = (
                    discovery_state.get("usage")
                    if isinstance(discovery_state, dict)
                    else None
                )
                if isinstance(usage, dict):
                    current_jev_usage.append(usage)
                return selected, fallback
            rail._route_deferred_tools = capture_routes

            missed_functions = test_entry.get("missed_function", {})
            for turn_index, messages in enumerate(test_entry.get("question", [])):
                if str(turn_index) in missed_functions:
                    additional = missed_functions[str(turn_index)]
                    benchmark_functions.extend(additional)
                    install_inventory(benchmark_functions)
                    if not messages:
                        from bfcl_eval.constants.default_prompts import (
                            DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
                        )
                        messages = [{
                            "role": "user",
                            "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
                        }]

                current_calls = []
                current_tokens = {"input": 0, "output": 0}
                current_jev_usage = []
                current_laya_usage = []
                detailed_model_usage: list[dict[str, Any]] = []
                summary_model_usage: list[dict[str, Any]] = []
                query = _message_text(messages)
                request = symbols["AgentRequest"](
                    request_id=uuid.uuid4().hex,
                    channel_id="web",
                    session_id=session_id,
                    req_method=symbols["ReqMethod"].CHAT_SEND,
                    params={
                        "query": query,
                        "mode": "agent",
                        "supports_user_interaction": False,
                    },
                )
                turn_started = time.perf_counter()
                output_chunks = []
                final_output = None
                async for chunk in swarm.process_message_stream(request):
                    payload = chunk.payload if isinstance(chunk.payload, dict) else {}
                    event_type = str(
                        payload.get("event_type") or payload.get("type") or ""
                    )
                    if event_type in {"chat.error", "error"}:
                        detail = payload.get("error") or payload.get("message") or payload
                        raise RuntimeError(f"JiuwenSwarm request failed: {detail}")
                    if event_type == "chat.usage_summary":
                        usage = payload.get("usage") or {}
                        record = _usage_record(usage, str(payload.get("model") or ""))
                        if record is not None:
                            summary_model_usage.append(record)
                    elif event_type in {"chat.usage_metadata", "llm_usage"}:
                        details = payload.get("metadata") or payload.get("payload")
                        if not isinstance(details, dict):
                            details = payload
                        raw_usage = details.get("usage_metadata", details)
                        record = _usage_record(raw_usage)
                        if record is not None:
                            detailed_model_usage.append(record)
                    if event_type in {"chat.final", "chat.delta"}:
                        content = payload.get("content")
                        if isinstance(content, str) and content:
                            if event_type == "chat.final":
                                final_output = content
                            else:
                                output_chunks.append(content)

                # Prefer per-model-call usage records when present. The final
                # usage_summary is cumulative in some runtimes and would
                # otherwise double-count those detailed records.
                agent_usage_records = detailed_model_usage or summary_model_usage
                deepseek_records = []
                other_agent_records = []
                for record in agent_usage_records:
                    if "deepseek" in str(record.get("model") or "").lower():
                        deepseek_records.append(record)
                    else:
                        other_agent_records.append(record)
                deepseek_usage = _sum_component_usage(deepseek_records)
                other_agent_usage = _sum_component_usage(other_agent_records)
                jev_usage = _sum_component_usage([
                    record
                    for raw in current_jev_usage
                    if (record := _usage_record(raw)) is not None
                ])
                laya_records = []
                for raw in current_laya_usage:
                    record = _usage_record(raw, "laya-local")
                    if record is not None:
                        # These are counts from Laya's own tokenizer and local
                        # inference, not provider/OpenRouter usage or cost.
                        record["cost_usd"] = None
                        laya_records.append(record)
                laya_local_usage = _sum_component_usage(laya_records)
                all_usage = _total_component_usage(
                    deepseek_usage, other_agent_usage, jev_usage
                )
                model_usage_by_turn.append({
                    "deepseek": deepseek_usage,
                    "jev": jev_usage,
                    "laya_local": laya_local_usage,
                    "other_agent_models": other_agent_usage,
                    "total": all_usage,
                })
                current_tokens["input"] = all_usage["input_tokens"]
                current_tokens["output"] = all_usage["output_tokens"]
                turn_latency = time.perf_counter() - turn_started
                calls_by_turn.append(current_calls)
                outputs_by_turn.append(final_output or "".join(output_chunks))
                token_input.append([current_tokens["input"]])
                token_output.append([current_tokens["output"]])
                latencies.append([turn_latency])
                if include_input_log:
                    inference_log.append({
                        "turn": turn_index,
                        "query": messages,
                        "selected_tools": selected_by_turn[-1] if selected_by_turn else [],
                        "tool_calls": current_calls,
                        "assistant_output": final_output or "".join(output_chunks),
                        "input_tokens": current_tokens["input"],
                        "output_tokens": current_tokens["output"],
                        "latency_seconds": turn_latency,
                    })

            if is_memory(category) and involved_classes:
                _, instances = execute_multi_turn_func_call(
                    [], initial_config, involved_classes, model_key, case_id,
                    long_context=long_context, is_evaL_run=False,
                )
                if len(instances) == 1:
                    flush = getattr(next(iter(instances.values())), "_flush_memory_to_local_file", None)
                    if callable(flush):
                        flush()

            metadata: dict[str, Any] = {}
            metadata["jiuwenswarm"] = {
                "tool_discovery_backend": discovery,
                "selected_tools": selected_by_turn,
                "fallback_to_bm25": fallback_by_turn,
            }
            component_names = (
                "deepseek",
                "jev",
                "laya_local",
                "other_agent_models",
                "total",
            )
            conversation_model_usage = {
                component: _total_component_usage(*[
                    turn_usage[component] for turn_usage in model_usage_by_turn
                ])
                for component in component_names
            }
            metadata["model_usage_by_turn"] = model_usage_by_turn
            metadata["conversation_model_usage"] = conversation_model_usage
            metadata["conversation_totals"] = {
                "latency_seconds": sum(item[0] for item in latencies),
                "input_tokens": sum(item[0] for item in token_input),
                "output_tokens": sum(item[0] for item in token_output),
                "total_tokens": sum(
                    item[0] for item in token_input
                ) + sum(item[0] for item in token_output),
                "model_usage": conversation_model_usage,
            }
            if include_input_log:
                metadata["inference_log"] = inference_log
            if multi_turn:
                result = []
                for turn_index, calls in enumerate(calls_by_turn):
                    turn_result = [calls] if calls else []
                    output_text = outputs_by_turn[turn_index]
                    if output_text:
                        turn_result.append(output_text)
                    result.append(turn_result)
                turn_metrics = []
                for index, (input_count, output_count, latency) in enumerate(
                    zip(token_input, token_output, latencies), start=1
                ):
                    input_total = sum(input_count)
                    output_total = sum(output_count)
                    turn_metrics.append({
                        "turn": index,
                        "latency_seconds": sum(latency),
                        "input_tokens": input_total,
                        "output_tokens": output_total,
                        "total_tokens": input_total + output_total,
                        "model_usage": model_usage_by_turn[index - 1],
                    })
                metadata["turn_metrics"] = turn_metrics
                metadata["input_token_count"] = token_input
                metadata["output_token_count"] = token_output
                metadata["latency"] = latencies
            else:
                result = calls_by_turn[0] if calls_by_turn else []
                metadata["input_token_count"] = sum(
                    turn[0] for turn in token_input
                )
                metadata["output_token_count"] = sum(
                    turn[0] for turn in token_output
                )
                metadata["latency"] = latencies[0][0] if latencies else 0.0
            return result, metadata
        finally:
            cleanup = getattr(root_adapter, "cleanup_session_adapter", None)
            if callable(cleanup):
                try:
                    await cleanup(session_id)
                except Exception:
                    pass
            await swarm.cleanup()

    def decode_ast(self, result, language, has_tool_call_tag):
        if isinstance(result, dict):
            result = [result]
        decoded = []
        for call in result or []:
            if not isinstance(call, dict):
                continue
            for name, arguments in call.items():
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                decoded.append({name: arguments})
        return decoded

    def decode_execute(self, result, has_tool_call_tag):
        if isinstance(result, dict):
            result = [result]
        decoded = []
        for call in result or []:
            for name, arguments in call.items():
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                encoded = ",".join(
                    f"{key}={value!r}" for key, value in arguments.items()
                )
                decoded.append(f"{name}({encoded})")
        return decoded
