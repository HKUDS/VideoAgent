"""Focused SDK transport and real graph-dispatch tests; no LLM/GPU required."""

import asyncio
from contextlib import contextmanager
from functools import partial
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from mcp.client.streamable_http import streamablehttp_client

from environment.agents.base import FunctionRegistry
from environment.agents.multi import MultiAgent
from environment.roles import get_agent_class
from environment.roles.vid_news import news_story_gen
from environment.roles.web_researcher import MCP_URL, USER_AGENT

SOURCE = "https://www.nasa.gov/mission/artemis-ii/"
EXCERPT = f"Artemis II mission overview. Source: {SOURCE}"


class WireServer:
    """An HTTP boundary fixture; the real MCP SDK handles every RPC."""

    def __init__(self, error=False, empty=False, sse=False, missing_tool=False, slow=False):
        self.requests = []
        self.error = error
        self.empty = empty
        self.sse = sse
        self.missing_tool = missing_tool
        self.slow = slow
        self.called = asyncio.Event()
        self.cleaned = False

    async def handle(self, request):
        body = json.loads(request.content)
        self.requests.append((request, body))
        if "id" not in body:
            return httpx.Response(202)
        method = body["method"]
        if method == "initialize":
            result = {
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": [] if self.missing_tool else [{
                "name": "web_search", "description": "Search public sources",
                "inputSchema": {"type": "object"},
            }]}
        elif method == "tools/call":
            self.called.set()
            if self.slow:
                try:
                    await asyncio.sleep(10)
                finally:
                    self.cleaned = True
            result = {
                "isError": self.error,
                "content": [] if self.empty else [{
                    "type": "text", "text": "quota reached" if self.error else EXCERPT,
                }],
            }
        else:
            raise AssertionError(f"Unexpected RPC: {method}")
        payload = {"jsonrpc": "2.0", "id": body["id"], "result": result}
        if self.sse:
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"},
                                  content=f"event: message\ndata: {json.dumps(payload)}\n\n")
        return httpx.Response(200, json=payload)

    @contextmanager
    def connect(self):
        def factory(headers=None, timeout=None, auth=None):
            return httpx.AsyncClient(headers=headers, timeout=timeout, auth=auth,
                                     transport=httpx.MockTransport(self.handle))
        with patch("mcp.client.streamable_http.streamablehttp_client",
                   partial(streamablehttp_client, httpx_client_factory=factory)):
            yield


class WebResearchTests(unittest.TestCase):
    def setUp(self):
        self.role = get_agent_class("WebResearcher")()
        self.settings = patch.dict("environment.roles.web_researcher.config", {}, clear=True)
        self.settings.start()
        self.addCleanup(self.settings.stop)

    def execute(self):
        return self.role.execute(reqs="Create a short news narration", query="NASA Artemis II mission overview")

    def test_real_loader_transport_and_original_requirements(self):
        server = WireServer()
        with server.connect(), patch.dict("os.environ", {"PARALLEL_API_KEY": "unused-saved-key"}):
            result = self.execute()
        self.assertIn(SOURCE, result["researched_reqs"])
        self.assertIn("Create a short news narration", result["researched_reqs"])
        methods = [body["method"] for _, body in server.requests]
        self.assertEqual(methods, ["initialize", "notifications/initialized", "tools/list", "tools/call"])
        for request, body in server.requests:
            self.assertEqual(str(request.url), MCP_URL)
            self.assertEqual(request.headers["User-Agent"], USER_AGENT)
            self.assertNotIn("Authorization", request.headers)
            self.assertNotIn("Create a short news narration", request.content.decode())
        call = server.requests[-1][1]["params"]
        self.assertEqual(call["name"], "web_search")
        self.assertEqual(call["arguments"]["search_queries"], ["NASA Artemis II mission overview"])
        self.assertEqual(call["arguments"]["session_id"], self.role.session_id)

    def test_sse_response(self):
        with WireServer(sse=True).connect():
            self.assertIn(SOURCE, self.execute()["researched_reqs"])

    def test_disabled_search_never_opens_transport(self):
        with patch.dict("environment.roles.web_researcher.config", {"web_search": {"enabled": False}}), \
             patch("mcp.client.streamable_http.streamablehttp_client") as connect:
            with self.assertRaises(PermissionError):
                self.execute()
            connect.assert_not_called()

    def test_explicit_timeout_preserved(self):
        server = WireServer(slow=True)
        with patch.dict("environment.roles.web_researcher.config", {"web_search": {"timeout_seconds": 0.05}}), \
             server.connect():
            with self.assertRaisesRegex(TimeoutError, "timed out"):
                self.execute()
        self.assertTrue(server.cleaned)
        self.assertEqual(server.requests[-1][0].extensions["timeout"]["read"], 0.05)

    def test_cancellation_propagates_and_cleans_up(self):
        server = WireServer(slow=True)
        async def cancel_request():
            task = asyncio.create_task(self.role._bounded_search("NASA Artemis II mission overview", 5))
            await asyncio.wait_for(server.called.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        with server.connect():
            asyncio.run(cancel_request())
        self.assertTrue(server.cleaned)

    def test_errors_empty_results_and_missing_tool_stop_the_chain(self):
        for options, expected in (
            ({"error": True}, "Parallel web search failed: quota reached"),
            ({"empty": True}, "Parallel web search returned no text"),
            ({"missing_tool": True}, "Parallel MCP did not advertise web_search"),
        ):
            with self.subTest(options=options), WireServer(**options).connect():
                # AnyIO context managers may wrap the original RuntimeError in
                # an ExceptionGroup; both must escape, rather than return prose.
                with self.assertRaises(Exception) as caught:
                    self.execute()
                def messages(exc):
                    return str(exc) + " ".join(messages(child) for child in getattr(exc, "exceptions", ()))
                self.assertIn(expected, messages(caught.exception))

    def test_invalid_settings_and_blank_query_make_no_request(self):
        with patch("mcp.client.streamable_http.streamablehttp_client") as connect:
            for query in ("", "   "):
                with self.assertRaises(ValueError):
                    self.role.execute(reqs="Write news", query=query)
            with patch.dict("environment.roles.web_researcher.config", {"web_search": {"enabled": "false"}}):
                with self.assertRaises(ValueError):
                    self.execute()
            connect.assert_not_called()

    def test_registry_intent_and_graph_feed_actual_news_presenter(self):
        # Script only the model responses. Registry resolution, graph execution,
        # HTTP MCP transport and NewsContentGenerator.execute are real.
        FunctionRegistry.register(type(self.role))
        router = MultiAgent.__new__(MultiAgent)
        router.registry = json.dumps(FunctionRegistry._registry)
        router.intents = router.load_yml("environment/config/intents.yml")
        self.assertIn("WebResearcher", router.get_tools_by_intents(["Web Research"]))
        graph = [
            {"node": "WebResearcher", "inputs": [{"name": "reqs"}, {"name": "query"}],
             "outputs": [{"name": "researched_reqs", "links": [{"NewsContentGenerator": "reqs"}]}]},
            {"node": "NewsContentGenerator", "inputs": [{"name": "reqs"}, {"name": "video_path"}, {"name": "news_present_style"}],
             "outputs": [{"name": "video_scene_path", "links": []}]},
        ]
        user_graph = [
            {"node": "requirements", "description": "", "links": [{"WebResearcher": "reqs"}]},
            {"node": "query", "description": "", "links": [{"WebResearcher": "query"}]},
            {"node": "video", "description": "", "links": [{"NewsContentGenerator": "video_path"}]},
            {"node": "style", "description": "", "links": [{"NewsContentGenerator": "news_present_style"}]},
        ]
        prompts = []
        def model(**kwargs):
            prompts.append(kwargs["user"])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="/////\nArtemis mission context."))])
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "transcript.lab").write_text("Reference video transcript.")
            with WireServer().connect(), patch("builtins.input", side_effect=[
                "Write a short factual news video", "NASA Artemis II mission overview", tmp, "Third-person narration",
            ]), patch.object(news_story_gen, "gpt", side_effect=model), \
                 patch.object(news_story_gen.os, "getcwd", return_value=tmp):
                context = router.execute_agent_chain(graph, ["WebResearcher", "NewsContentGenerator"], user_graph)
            self.assertIn(SOURCE, context["NewsContentGenerator.reqs"])
            self.assertIn(SOURCE, prompts[0])
            self.assertIn("Reference video transcript.", prompts[0])
            output = json.loads(Path(tmp, "dataset/video_edit/scene_output/video_scene.json").read_text())
            self.assertIn(SOURCE, output["user_idea"])
            self.assertIn("Artemis mission context", output["content_created"])


if __name__ == "__main__":
    unittest.main()
