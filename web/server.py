#!/usr/bin/env python3
"""Lightweight web server for interactive RLM tracing."""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent to path for rlm imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aiohttp import web
import aiohttp

from rlm.config import load_config
from rlm.cache.store import CacheStore
from rlm.evaluator.lightweight import LightweightEvaluator
from rlm.llm.client import LLMClient
from rlm.llm.parser import parse_llm_output, ParseError
from rlm.llm.prompts import SYSTEM_PROMPT
from rlm.types import ExploreAction, CommitPlan, FinalAnswer, OpType

# Store active sessions
sessions = {}


async def index_handler(request):
    """Serve the main HTML page."""
    html_path = Path(__file__).parent / "index.html"
    return web.FileResponse(html_path)


async def sample_handler(request):
    """Serve sample data."""
    sample_type = request.match_info["type"]
    data_dir = Path(__file__).parent.parent / "data"
    
    if sample_type == "needle":
        context_file = data_dir / "needle_context.txt"
        if context_file.exists():
            return web.json_response({
                "context": context_file.read_text()
            })
    elif sample_type == "codeqa":
        context_file = data_dir / "codeqa_context.txt"
        question_file = data_dir / "codeqa_question.json"
        if context_file.exists() and question_file.exists():
            q = json.loads(question_file.read_text())
            query = f"{q['question']}\n\nA: {q['choice_A']}\nB: {q['choice_B']}\nC: {q['choice_C']}\nD: {q['choice_D']}\n\nAnswer with ONLY the letter."
            return web.json_response({
                "query": query,
                "context": context_file.read_text()
            })
    
    return web.json_response({"error": "Sample not found"}, status=404)


async def websocket_handler(request):
    """Handle WebSocket connections for interactive tracing."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    session_id = id(ws)
    sessions[session_id] = {"ws": ws, "running": False, "pending_event": None}
    
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                await handle_ws_message(session_id, data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f"WebSocket error: {ws.exception()}")
    finally:
        sessions.pop(session_id, None)
    
    return ws


async def handle_ws_message(session_id, data):
    """Handle incoming WebSocket messages."""
    session = sessions.get(session_id)
    if not session:
        return
    
    ws = session["ws"]
    msg_type = data.get("type")
    
    if msg_type == "start":
        session["running"] = True
        asyncio.create_task(run_trace(session_id, data))
    
    elif msg_type == "approve":
        if session.get("pending_event"):
            session["approved_action"] = data.get("action")
            session["pending_event"].set()
    
    elif msg_type == "skip":
        if session.get("pending_event"):
            session["skip"] = True
            session["pending_event"].set()
    
    elif msg_type == "stop":
        session["running"] = False
        if session.get("pending_event"):
            session["pending_event"].set()


async def run_trace(session_id, params):
    """Run the RLM trace with interactive approval."""
    session = sessions.get(session_id)
    if not session:
        return
    
    ws = session["ws"]
    
    try:
        model = params.get("model", "gpt-4o-mini")
        max_depth = params.get("maxDepth", 1)
        query = params["query"]
        context = params["context"]
        
        config = load_config(
            model=model,
            max_recursion_depth=max_depth,
            verbose=False
        )
        
        client = LLMClient(config)
        cache = CacheStore(config.cache_dir)
        evaluator = LightweightEvaluator(cache=cache)
        
        # Build system prompt
        system_prompt = SYSTEM_PROMPT.format(
            context_chars=f"{len(context):,}",
            query=query
        )
        
        client.set_system_prompt(system_prompt)
        bindings = {"context": context}
        
        explore_steps = 0
        commit_cycles = 0
        max_explore = config.max_explore_steps
        total_tokens = 0
        
        # Initial message
        response = await asyncio.to_thread(
            client.send, "Begin. The context variable is available."
        )
        
        input_tokens, output_tokens = client.get_token_usage()
        total_tokens = input_tokens + output_tokens
        await ws.send_json({"type": "tokens", "count": total_tokens})
        
        while session.get("running", False):
            # Parse the response
            try:
                action = parse_llm_output(response)
            except ParseError as e:
                await ws.send_json({
                    "type": "error",
                    "error": str(e)
                })
                # Try to recover
                response = await asyncio.to_thread(
                    client.send,
                    f"Your response was not valid JSON. Please respond with a valid JSON "
                    f"object with a 'mode' field. Error: {e}"
                )
                continue
            
            # Determine action type and summary
            if isinstance(action, ExploreAction):
                mode = "explore"
                summary = f"{action.operation.op.value}({json.dumps(action.operation.args)[:50]})"
            elif isinstance(action, CommitPlan):
                mode = "commit"
                ops_summary = ", ".join(op.op.value for op in action.operations)
                summary = f"{len(action.operations)} ops: {ops_summary[:50]}"
            elif isinstance(action, FinalAnswer):
                mode = "final"
                summary = "Answer ready"
            else:
                mode = "unknown"
                summary = ""
            
            # Send step to client
            await ws.send_json({
                "type": "step",
                "mode": mode,
                "summary": summary,
                "action": action_to_dict(action),
                "raw": response[:1000]
            })
            
            # Wait for approval
            session["pending_event"] = asyncio.Event()
            session["skip"] = False
            session["approved_action"] = None
            
            await ws.send_json({
                "type": "pending",
                "action": action_to_dict(action)
            })
            
            await session["pending_event"].wait()
            
            if not session.get("running", False):
                break
            
            if session.get("skip"):
                response = await asyncio.to_thread(
                    client.send, "Skipped. Try a different approach."
                )
                continue
            
            # Use approved action (possibly edited)
            if session.get("approved_action"):
                action = dict_to_action(session["approved_action"])
            
            # Handle the action
            if isinstance(action, FinalAnswer):
                await ws.send_json({
                    "type": "final",
                    "answer": action.answer
                })
                break
            
            elif isinstance(action, ExploreAction):
                explore_steps += 1
                if explore_steps > max_explore:
                    response = await asyncio.to_thread(
                        client.send,
                        f"You have reached the maximum of {max_explore} explore steps. "
                        f"Please COMMIT a plan or provide a FINAL answer."
                    )
                    continue
                
                # Execute the operation
                try:
                    result = evaluator.execute(action.operation, bindings)
                    if action.operation.bind:
                        bindings[action.operation.bind] = result.value
                    
                    # Truncate result for display
                    result_str = result.value
                    if len(result_str) > 4000:
                        result_str = result_str[:4000] + f"... [{len(result.value)} chars total]"
                    
                    await ws.send_json({
                        "type": "result",
                        "result": result_str[:2000]
                    })
                    
                    response = await asyncio.to_thread(
                        client.send,
                        f"Result of {action.operation.op.value}:\n{result_str}"
                    )
                    
                    input_tokens, output_tokens = client.get_token_usage()
                    await ws.send_json({"type": "tokens", "count": input_tokens + output_tokens})
                    
                except Exception as e:
                    await ws.send_json({
                        "type": "error",
                        "error": f"Operation failed: {e}"
                    })
                    response = await asyncio.to_thread(
                        client.send, f"Error executing {action.operation.op.value}: {e}"
                    )
            
            elif isinstance(action, CommitPlan):
                commit_cycles += 1
                
                # Execute all operations in the plan
                try:
                    for i, op in enumerate(action.operations):
                        await ws.send_json({
                            "type": "step",
                            "mode": "commit",
                            "summary": f"Op {i+1}/{len(action.operations)}: {op.op.value}",
                            "action": {"op": op.op.value, "args": op.args, "bind": op.bind}
                        })
                        
                        if op.op == OpType.RLM_CALL:
                            # Recursive call
                            query_text = op.args.get("query", "")
                            ctx_ref = op.args.get("context", "")
                            ctx_text = bindings.get(ctx_ref, "")
                            
                            from rlm.orchestrator import RLMOrchestrator
                            sub_config = load_config(
                                model=model,
                                max_recursion_depth=max_depth - 1,
                                verbose=False
                            )
                            sub_orch = RLMOrchestrator(sub_config)
                            result_value = await asyncio.to_thread(
                                sub_orch.run, query_text, ctx_text, depth=1
                            )
                            
                            sub_in, sub_out = sub_orch.get_total_token_usage()
                            await ws.send_json({"type": "tokens", "count": sub_in + sub_out})
                            
                        elif op.op == OpType.MAP:
                            # Parallel map
                            prompt = op.args.get("prompt", "")
                            input_ref = op.args.get("input", "")
                            raw = bindings.get(input_ref, "[]")
                            items = json.loads(raw) if raw.startswith("[") else [raw]
                            
                            results = []
                            for j, item in enumerate(items):
                                await ws.send_json({
                                    "type": "step",
                                    "mode": "commit",
                                    "summary": f"MAP {j+1}/{len(items)}",
                                    "action": {"op": "map", "chunk": j}
                                })
                                
                                from rlm.orchestrator import RLMOrchestrator
                                sub_config = load_config(
                                    model=model,
                                    max_recursion_depth=max_depth - 1,
                                    verbose=False
                                )
                                sub_orch = RLMOrchestrator(sub_config)
                                r = await asyncio.to_thread(
                                    sub_orch.run, prompt, str(item), depth=1
                                )
                                results.append(r)
                                
                                sub_in, sub_out = sub_orch.get_total_token_usage()
                                await ws.send_json({"type": "tokens", "count": sub_in + sub_out})
                            
                            result_value = json.dumps(results)
                        
                        else:
                            # Simple operations
                            result = evaluator.execute(op, bindings)
                            result_value = result.value
                        
                        if op.bind:
                            bindings[op.bind] = result_value
                        
                        if not session.get("running", False):
                            break
                    
                    # Get final result
                    final_result = bindings.get(action.output, "No output")
                    
                    await ws.send_json({
                        "type": "result",
                        "result": str(final_result)[:2000]
                    })
                    
                    response = await asyncio.to_thread(
                        client.send,
                        f"Commit plan executed. Result ({action.output}):\n{str(final_result)[:4000]}"
                    )
                    
                    input_tokens, output_tokens = client.get_token_usage()
                    await ws.send_json({"type": "tokens", "count": input_tokens + output_tokens})
                    
                except Exception as e:
                    import traceback
                    await ws.send_json({
                        "type": "error",
                        "error": f"Commit failed: {e}\n{traceback.format_exc()}"
                    })
                    response = await asyncio.to_thread(
                        client.send, f"Error executing commit plan: {e}"
                    )
    
    except Exception as e:
        import traceback
        await ws.send_json({
            "type": "error",
            "error": f"{e}\n{traceback.format_exc()}"
        })
    
    finally:
        session["running"] = False


def action_to_dict(action):
    """Convert an action to a JSON-serializable dict."""
    if isinstance(action, ExploreAction):
        return {
            "mode": "explore",
            "operation": {
                "op": action.operation.op.value,
                "args": action.operation.args,
                "bind": action.operation.bind
            }
        }
    elif isinstance(action, CommitPlan):
        return {
            "mode": "commit",
            "operations": [
                {
                    "op": op.op.value,
                    "args": op.args,
                    "bind": op.bind
                }
                for op in action.operations
            ],
            "output": action.output
        }
    elif isinstance(action, FinalAnswer):
        return {
            "mode": "final",
            "answer": action.answer
        }
    return {}


def dict_to_action(d):
    """Convert a dict back to an action object."""
    from rlm.types import Operation, OpType
    
    mode = d.get("mode")
    if mode == "explore":
        op_data = d["operation"]
        return ExploreAction(
            operation=Operation(
                op=OpType(op_data["op"]),
                args=op_data.get("args", {}),
                bind=op_data.get("bind")
            )
        )
    elif mode == "commit":
        operations = [
            Operation(
                op=OpType(op_data["op"]),
                args=op_data.get("args", {}),
                bind=op_data.get("bind")
            )
            for op_data in d["operations"]
        ]
        return CommitPlan(operations=operations, output=d.get("output", "result"))
    elif mode == "final":
        return FinalAnswer(answer=d["answer"])
    return None


def main():
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/api/sample/{type}", sample_handler)
    app.router.add_get("/ws", websocket_handler)
    
    port = int(os.environ.get("PORT", 8765))
    print(f"Starting RLM Tracer on http://localhost:{port}")
    web.run_app(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
