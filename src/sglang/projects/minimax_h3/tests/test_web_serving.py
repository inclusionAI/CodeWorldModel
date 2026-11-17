import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from aiohttp import ClientSession, WSMsgType, WSServerHandshakeError, web

from tools.serve_web import bootstrap

bootstrap()

from projects.minimax_h3.inference.serving_http import pack, unpack
from projects.minimax_h3.inference.web_serving import (
    GATEWAY, SESSION_COOKIE, FixtureBackend, create_app, profiling_status, read_public_access,
    source_budget_capability)
from projects.minimax_h3.modeling.ref_recipe import FIXED_STREAMING_TEXT_RECIPE


class ControlledFixture(FixtureBackend):
    def __init__(self):
        super().__init__(width=2, height=1)
        self.entered = asyncio.Event()
        self.gate = None
        self.delay_first = False
        self.corrupt = None
        self.cancel_fails = False
        self.cancelled = 0
        self.reply_fields = {}

    async def render(self, job):
        if job.op == 'cancel':
            self.cancelled += 1
        if job.op != 'chunk':
            return b''
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        rgb = await super().render(job)
        if self.delay_first and job.metadata['chunk_id'] == 0:
            self.pending = rgb
            return b''
        rgb, self.pending = self.pending + rgb, b''
        return rgb

    async def handle(self, request):
        if self.cancel_fails and request.path.endswith('/cancel'):
            return web.Response(status=503, body=pack(dict(error='test cancel failed')))
        response = await super().handle(request)
        if request.method == 'POST' and self.reply_fields:
            metadata, rgb = unpack(response.body)
            response.body = pack(dict(metadata, **self.reply_fields), rgb)
        if request.path.endswith('/chunk') and self.corrupt:
            metadata, rgb = unpack(response.body)
            if self.corrupt == 'source':
                metadata['source_frames'][0]['source_frame'] += 1
            elif self.corrupt == 'count':
                metadata['frame_count'] += 1
            elif self.corrupt == 'geometry':
                metadata['width'] += 1
            elif self.corrupt == 'pixels':
                rgb += b'x'
            elif self.corrupt == 'boolean':
                metadata['source_frames'][0]['source_frame'] = False
            response.body = pack(metadata, rgb)
        return response


class WebServingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = ControlledFixture()
        self.app = await create_app(fixture=True, fixture_backend=self.fixture, cleanup_timeout=.05)
        self.gateway = self.app[GATEWAY]
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        await web.TCPSite(self.runner, '127.0.0.1', 0).start()
        self.url = 'http://127.0.0.1:' + str(self.runner.addresses[0][1])
        self.client = ClientSession()

    async def asyncTearDown(self):
        if self.fixture.gate:
            self.fixture.gate.set()
        await self.client.close()
        await self.runner.cleanup()

    async def connect(self):
        return await self.client.ws_connect(self.url + '/stream', origin=self.url)

    async def receive(self, socket):
        message = await socket.receive(timeout=3)
        self.assertEqual(message.type, WSMsgType.BINARY, message)
        return unpack(message.data)

    async def start(self, socket):
        await socket.send_bytes(pack(dict(op='start', seed=2026, condition_seed=2026), bytes(6)))
        metadata, pixels = await self.receive(socket)
        self.assertEqual(metadata['op'], 'started')
        self.assertEqual(pixels, b'')
        return metadata

    def chunk(self, index=0, eos=False, **overrides):
        first = index*17
        count = 5 if eos else 17
        metadata = dict(op='chunk', chunk_id=index, first_source_frame=first,
                        capture_ns=list(range(first, first+count)), assembled_ns=first+count, eos=eos)
        metadata.update(overrides)
        return pack(metadata, bytes([index+1])*count*6)

    async def released(self):
        async with asyncio.timeout(3):
            while self.gateway.owner is not None:
                await asyncio.sleep(.01)

    async def test_racing_source_is_opt_in_and_serves_only_build_assets(self):
        self.assertNotIn('torcs_source', self.gateway.config)
        async with self.client.get(self.url + '/threejs-racing/index.html') as response:
            self.assertEqual(response.status, 404)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'web' / 'playable'
            build = root.parent / 'threejs-racing' / 'dist-streaming'
            (build / 'assets').mkdir(parents=True)
            (build / 'torcs' / 'textures').mkdir(parents=True)
            for name in ('index.html', 'palette.json', 'assets/game.js', 'torcs/textures/car.png', '.env', 'private.json'):
                (build / name).write_text('test', encoding='utf-8')
            with patch('projects.minimax_h3.inference.web_serving.ROOT', root):
                app = await create_app(fixture=True, fixture_backend=FixtureBackend(width=2, height=1), threejs_racing=True)
            runner = web.AppRunner(app)
            try:
                await runner.setup()
                await web.TCPSite(runner, '127.0.0.1', 0).start()
                url = 'http://127.0.0.1:' + str(runner.addresses[0][1])
                async with self.client.get(url + '/config') as response:
                    self.assertEqual((await response.json())['torcs_source']['style'], 'racing-proxy/v1')
                for name in ('index.html', 'palette.json', 'assets/game.js', 'torcs/textures/car.png'):
                    async with self.client.get(url + '/threejs-racing/' + name) as response:
                        self.assertEqual(response.status, 200, name)
                for name in ('.env', 'private.json', '../.git/config', 'assets/../../private.json'):
                    async with self.client.get(url + '/threejs-racing/' + name) as response:
                        self.assertEqual(response.status, 404, name)
            finally:
                await runner.cleanup()

    async def test_racing_requires_boolean_and_built_entry(self):
        with self.assertRaisesRegex(ValueError, 'boolean'):
            await create_app(fixture=True, threejs_racing='yes')
        with tempfile.TemporaryDirectory() as directory:
            with patch('projects.minimax_h3.inference.web_serving.ROOT', Path(directory) / 'web' / 'playable'):
                with self.assertRaisesRegex(ValueError, 'build:streaming'):
                    await create_app(fixture=True, threejs_racing=True)
                with self.assertRaisesRegex(ValueError, 'build:streaming'):
                    await create_app(fixture=True, ballpoint_breach=True)

    async def test_ballpoint_source_is_opt_in_and_limits_assets(self):
        self.assertNotIn('ballpoint_source', self.gateway.config)
        async with self.client.get(self.url + '/ballpoint-breach/index.html') as response:
            self.assertEqual(response.status, 404)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            await create_app(fixture=True, ballpoint_breach='yes')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'web' / 'playable'
            build = root.parent / 'ballpoint-breach' / 'dist-streaming'
            (build / 'assets').mkdir(parents=True)
            for name in ('index.html', 'assets/game.js', 'assets/style.css', 'assets/.env', 'private.json'):
                (build / name).write_text('test', encoding='utf-8')
            with patch('projects.minimax_h3.inference.web_serving.ROOT', root):
                app = await create_app(fixture=True, fixture_backend=FixtureBackend(width=2, height=1), ballpoint_breach=True)
            runner = web.AppRunner(app)
            try:
                await runner.setup()
                await web.TCPSite(runner, '127.0.0.1', 0).start()
                url = 'http://127.0.0.1:' + str(runner.addresses[0][1])
                async with self.client.get(url + '/config') as response:
                    self.assertEqual((await response.json())['ballpoint_source']['style'], 'ballpoint-rgb/v1')
                for name in ('index.html', 'assets/game.js', 'assets/style.css'):
                    async with self.client.get(url + '/ballpoint-breach/' + name) as response:
                        self.assertEqual(response.status, 200, name)
                for name in ('assets/.env', 'private.json', 'assets/../../private.json'):
                    async with self.client.get(url + '/ballpoint-breach/' + name) as response:
                        self.assertEqual(response.status, 404, name)
            finally:
                await runner.cleanup()

    async def test_config_and_normal_eos(self):
        async with self.client.get(self.url+'/sources.mjs') as response:
            self.assertEqual(response.status, 200)
            self.assertIn('sourceScenes', await response.text())
        async with self.client.get(self.url+'/recording.mjs') as response:
            self.assertEqual(response.status, 200)
            self.assertIn('PairedRecorder', await response.text())
        async with self.client.get(self.url+'/config') as response:
            config = await response.json()
        self.assertEqual(config['mode'], 'fixture')
        self.assertEqual(config['fps'], 24)
        self.assertEqual(config['max_capture_frames'], 34)
        self.assertEqual(config['profiling'], dict(status='fixture', source='fixture'))
        self.assertEqual(config['source_budget'], dict(eligible=False, reason='requires_h264'))
        socket = await self.connect()
        started = await self.start(socket)
        self.assertEqual(started['profiling'], config['profiling'])
        self.assertEqual(started['source_budget'], config['source_budget'])
        for index, eos in ((0, False), (1, True)):
            await socket.send_bytes(self.chunk(index, eos))
            metadata, rgb = await self.receive(socket)
            count = 5 if eos else 17
            self.assertEqual(metadata['op'], 'output')
            self.assertEqual(metadata['frame_count'], count)
            self.assertEqual(metadata['source_frames'][0]['source_frame'], index*17)
            self.assertEqual(rgb, bytes([index+1])*count*6)
            self.assertGreaterEqual(metadata['rpc_ms'], 0)
            self.assertEqual(metadata['profiling'], config['profiling'])
        self.assertEqual(metadata['status'], 'complete')
        await socket.close()
        await self.released()
        self.assertEqual(self.fixture.cancelled, 0)

    async def test_raw_started_budget_uses_receipt_media(self):
        self.fixture.reply_fields['media'] = dict(protocol='h264_annexb_v1')
        socket = await self.connect()
        started = await self.start(socket)
        self.assertEqual(self.gateway.config['source_budget'], dict(eligible=False, reason='requires_h264'))
        self.assertEqual(started['source_budget'], dict(eligible=True, reason='fixture_h264'))
        await socket.close()

    def test_budget_numeric_recipe_fields_match_browser_numbers(self):
        config = dict(mode='model', media=dict(protocol='h264_annexb_v1'), codec=dict(decoder='causal'),
            recipe=dict(protocol='ref_fixed_recent_history/v2', conditioning_mode='static_image_qwen_proxy_vae_v1',
                        bootstrap_latents=9.0, current_latent_schedule=[5.0]))
        self.assertEqual(source_budget_capability(config), dict(eligible=True,
            reason='fixed_v2_bootstrap9_current5_causal'))

    def test_cwm_budget_preserves_five_latent_bootstrap(self):
        config = dict(mode='model', media=dict(protocol='h264_annexb_v1'), codec=dict(decoder='causal'),
            recipe=dict(protocol='cwm_causal/v1', conditioning_mode='static_image_qwen_proxy_vae_v1',
                        bootstrap_latents=5, current_latent_schedule=[5]))
        self.assertEqual(source_budget_capability(config),
                         dict(eligible=True, reason='cwm_chunk5_window6_causal'))
        for bootstrap in (9, 47):
            config['recipe']['bootstrap_latents'] = bootstrap
            self.assertEqual(source_budget_capability(config),
                             dict(eligible=False, reason='unsupported_recipe_or_decoder'))
        config['recipe']['bootstrap_latents'] = 5.0
        self.assertTrue(source_budget_capability(config)['eligible'])
        config['codec']['decoder'] = 'offline'
        self.assertFalse(source_budget_capability(config)['eligible'])

    def test_profile_requires_explicit_boolean(self):
        for value in (None, 0, 1, 'false', 'true', [], {}):
            metadata = dict(effective_settings=dict(deployment=dict(options=dict(profile=value))))
            self.assertEqual(profiling_status(metadata, fixture=False, source='health'),
                             dict(status='unknown', source='health'))
        for metadata in ({}, None, dict(effective_settings=[]),
                         dict(effective_settings=dict(deployment='invalid')),
                         dict(effective_settings=dict(deployment=dict(options=None)))):
            self.assertEqual(profiling_status(metadata, fixture=False, source='start'),
                             dict(status='unknown', source='start'))
        self.assertEqual(profiling_status(dict(effective_settings=dict(deployment=dict(options=dict(profile=True)))),
                                         fixture=True, source='health'), dict(status='fixture', source='fixture'))

    async def test_model_profiling_uses_start_receipt_and_preserves_traces(self):
        for health_value, start_value, expected in ((False, True, 'diagnostic'), (True, False, 'baseline'),
                                                    (False, None, 'unknown')):
            with self.subTest(health=health_value, start=start_value):
                backend = ControlledFixture()
                backend.health['effective_settings'] = dict(deployment=dict(options=dict(profile=health_value)))
                backend.reply_fields = dict(profile=dict(stages={'decode': dict(host_ms=4)}),
                                            rank_profiles=[dict(spans=[dict(stage='decode', host_ms=4)])])
                if start_value is not None:
                    backend.reply_fields['effective_settings'] = dict(deployment=dict(options=dict(profile=start_value)))
                backend_url = await backend.start()
                runner = None
                try:
                    app = await create_app(backend=backend_url, cleanup_timeout=.05)
                    runner = web.AppRunner(app)
                    await runner.setup()
                    await web.TCPSite(runner, '127.0.0.1', 0).start()
                    url = 'http://127.0.0.1:' + str(runner.addresses[0][1])
                    async with self.client.get(url+'/config') as response:
                        config = await response.json()
                    self.assertEqual(config['profiling'], dict(
                        status='diagnostic' if health_value else 'baseline', source='health'))
                    async with self.client.ws_connect(url+'/stream', origin=url) as socket:
                        started = await self.start(socket)
                        self.assertEqual(started['profiling'], dict(status=expected, source='start'))
                        self.assertEqual(started['profile'], backend.reply_fields['profile'])
                        self.assertEqual(started['rank_profiles'], backend.reply_fields['rank_profiles'])
                        backend.reply_fields.pop('effective_settings', None)
                        for index, eos in ((0, False), (1, True)):
                            await socket.send_bytes(self.chunk(index, eos))
                            metadata, _ = await self.receive(socket)
                            self.assertEqual(metadata['profiling'], started['profiling'])
                            self.assertEqual(metadata['profile'], backend.reply_fields['profile'])
                            self.assertEqual(metadata['rank_profiles'], backend.reply_fields['rank_profiles'])
                finally:
                    if runner is not None:
                        await runner.cleanup()
                    await backend.runner.cleanup()

    async def test_zero_then_delayed_outputs_keep_fifo(self):
        self.fixture.delay_first = True
        socket = await self.connect()
        await self.start(socket)
        await socket.send_bytes(self.chunk())
        metadata, rgb = await self.receive(socket)
        self.assertEqual(metadata['frame_count'], 0)
        self.assertEqual(metadata['credit'], 1)
        self.assertEqual(rgb, b'')
        await socket.send_bytes(self.chunk(1))
        metadata, rgb = await self.receive(socket)
        self.assertEqual(metadata['frame_count'], 34)
        self.assertEqual([row['source_frame'] for row in metadata['source_frames']], list(range(34)))
        self.assertEqual(rgb, bytes([1])*17*6 + bytes([2])*17*6)
        await socket.send_bytes(self.chunk(2, True))
        metadata, rgb = await self.receive(socket)
        self.assertEqual(metadata['total_frames'], 39)
        self.assertEqual(metadata['status'], 'complete')
        await socket.close()

    async def test_bad_inputs_fail_session(self):
        inputs = [self.chunk(1), self.chunk(capture_ns=list(range(16))),
                  self.chunk(assembled_ns=0), self.chunk(capture_ns=[1]*16+[0]),
                  self.chunk(eos='yes'), self.chunk()+b'x', b'bad',
                  pack(dict(op='start', seed=True, condition_seed=2026), bytes(6))]
        for payload in inputs:
            with self.subTest(payload=payload[:80]):
                socket = await self.connect()
                await self.start(socket)
                await socket.send_bytes(payload)
                metadata, _ = await self.receive(socket)
                self.assertEqual(metadata['op'], 'error')
                await socket.close()
                await self.released()

    async def test_plain_text_and_bad_reference_rejected(self):
        socket = await self.connect()
        await socket.send_str('{}')
        self.assertEqual((await self.receive(socket))[0]['op'], 'error')
        await socket.close()
        await self.released()
        socket = await self.connect()
        await socket.send_bytes(pack(dict(op='start', seed=2026, condition_seed=2026), b'x'))
        self.assertEqual((await self.receive(socket))[0]['op'], 'error')
        await socket.close()
        await self.released()
        self.assertEqual(self.fixture.cancelled, 0)

    async def test_host_origin_and_static_paths(self):
        for headers in ({'Host':'evil.example'}, {'Origin':'https://evil.example'},
                        {'Origin':'null'}, {'Origin':'http://localhost:1'}):
            async with self.client.get(self.url+'/config', headers=headers) as response:
                self.assertEqual(response.status, 403)
        for path in ('/charts.mjs', '/gantt.mjs', '/gantt-replay.mjs', '/waits.mjs',
                     '/boxing-scene.js', '/boxing-logic.mjs', '/racing-scene.js', '/racing-logic.mjs',
                     '/vendor/echarts-6.1.0/echarts.min.js',
                     '/vendor/echarts-6.1.0/LICENSE', '/vendor/echarts-6.1.0/NOTICE'):
            async with self.client.get(self.url+path) as response:
                self.assertEqual(response.status, 200)
                self.assertGreater(len(await response.read()), 0)
        for path in ('/README.md', '/vendor/secret', '/../../AGENTS.md'):
            async with self.client.get(self.url+path) as response:
                self.assertEqual(response.status, 404)

    async def test_public_proxy_requires_host_and_websocket_origin(self):
        origin = 'https://gateway.example.com'
        public_url = origin + '/proxy/instance_1:8765/'
        key = 'test-only-access-key-' * 3
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        key_path = Path(directory.name) / 'access-key'
        key_path.write_text(key)
        app = await create_app(fixture=True, fixture_backend=ControlledFixture(),
                               public_url=public_url, access_key_file=key_path)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, '127.0.0.1', 0).start()
        url = 'http://127.0.0.1:' + str(runner.addresses[0][1])
        headers = {'Host': 'gateway.example.com'}
        try:
            async with self.client.get(url + '/', headers=headers, allow_redirects=False) as response:
                self.assertEqual(response.status, 303)
                self.assertEqual(response.headers['Location'], 'login')
            for path in ('/config', '/app.js', '/style.css'):
                async with self.client.get(url + path, headers=headers) as response:
                    self.assertEqual(response.status, 401)
                    self.assertEqual(response.headers['Cache-Control'], 'no-store')
            async with self.client.get(url + '/login', headers=headers) as response:
                self.assertEqual(response.status, 200)
                body = await response.text()
                self.assertIn('type="password"', body)
                self.assertNotIn(key, body)
            for rejected in ({}, {'Host': 'evil.example', 'X-Forwarded-Host': 'gateway.example.com'},
                             dict(headers, Origin='https://evil.example'), dict(headers, Origin='null')):
                async with self.client.get(url + '/config', headers=rejected) as response:
                    self.assertEqual(response.status, 403)
            for rejected in (None, 'http://gateway.example.com', 'https://evil.example', 'null'):
                with self.assertRaises(WSServerHandshakeError) as caught:
                    await self.client.ws_connect(url + '/stream', headers=headers, origin=rejected)
                self.assertEqual(caught.exception.status, 403)
            with self.assertRaises(WSServerHandshakeError) as caught:
                await self.client.ws_connect(url + '/stream', headers=headers, origin=origin)
            self.assertEqual(caught.exception.status, 401)
            async with self.client.post(url + '/login', headers=headers, data={'password': key}) as response:
                self.assertEqual(response.status, 403, 'login POST requires the exact Origin')
            login_headers = dict(headers, Origin=origin)
            async with self.client.post(url + '/login', headers=login_headers,
                                        data={'password': 'incorrect'}) as response:
                self.assertEqual(response.status, 401)
                self.assertNotIn('Set-Cookie', response.headers)
            async with self.client.post(url + '/login', headers=login_headers,
                                        data={'password': 'x' * 5000}) as response:
                self.assertEqual(response.status, 413)
            async with self.client.post(url + '/login', headers=login_headers,
                                        data={'password': key}, allow_redirects=False) as response:
                self.assertEqual(response.status, 303)
                self.assertEqual(response.headers['Location'], './')
                cookie = response.cookies[SESSION_COOKIE]
                self.assertEqual(cookie['path'], '/proxy/instance_1:8765/')
                self.assertTrue(cookie['secure'])
                self.assertTrue(cookie['httponly'])
                self.assertEqual(cookie['samesite'], 'Strict')
                self.assertEqual(cookie['domain'], '')
                self.assertNotEqual(cookie.value, key)
            signed_in = dict(headers, Cookie=f'{SESSION_COOKIE}={cookie.value}')
            async with self.client.get(url + '/config', headers=signed_in) as response:
                self.assertEqual(response.status, 200)
            async with self.client.get(url + '/config', headers=dict(headers,
                                        Cookie=f'{SESSION_COOKIE}=wrong-token')) as response:
                self.assertEqual(response.status, 401)
            async with self.client.ws_connect(url + '/stream', headers=signed_in, origin=origin) as socket:
                await self.start(socket)
                await socket.send_bytes(self.chunk(eos=True))
                metadata, rgb = await self.receive(socket)
                self.assertEqual(metadata['status'], 'complete')
                self.assertEqual(rgb, bytes([1]) * 5 * 6)
            for path in ('/README.md', '/session', '/health', '/private.png'):
                async with self.client.get(url + path, headers=signed_in) as response:
                    self.assertEqual(response.status, 404)
        finally:
            await runner.cleanup()

        restarted = await create_app(fixture=True, fixture_backend=ControlledFixture(),
                                     public_url=public_url, access_key_file=key_path)
        runner = web.AppRunner(restarted)
        await runner.setup()
        await web.TCPSite(runner, '127.0.0.1', 0).start()
        url = 'http://127.0.0.1:' + str(runner.addresses[0][1])
        try:
            async with self.client.get(url + '/config', headers=signed_in) as response:
                self.assertEqual(response.status, 401, 'old cookie is rejected after gateway restart')
        finally:
            await runner.cleanup()

    async def test_busy_and_disconnect_cancels(self):
        socket = await self.connect()
        await self.start(socket)
        with self.assertRaises(WSServerHandshakeError) as caught:
            await self.connect()
        self.assertEqual(caught.exception.status, 409)
        await socket.close()
        await self.released()
        self.assertEqual(self.fixture.cancelled, 1)
        replacement = await self.connect()
        await replacement.close()

    async def test_cancel_suppresses_late_output_and_holds_slot(self):
        self.fixture.gate = asyncio.Event()
        socket = await self.connect()
        await self.start(socket)
        await socket.send_bytes(self.chunk())
        await self.fixture.entered.wait()
        await socket.send_str(json.dumps(dict(op='cancel')))
        await asyncio.sleep(.08)
        with self.assertRaises(WSServerHandshakeError) as caught:
            await self.connect()
        self.assertEqual(caught.exception.status, 409)
        self.fixture.gate.set()
        self.assertEqual((await self.receive(socket))[0]['op'], 'cancelled')
        await socket.close()
        await self.released()
        self.assertEqual(self.fixture.cancelled, 1)

    async def test_second_inflight_request_fails_without_queue(self):
        self.fixture.gate = asyncio.Event()
        socket = await self.connect()
        await self.start(socket)
        await socket.send_bytes(self.chunk())
        await self.fixture.entered.wait()
        await socket.send_bytes(self.chunk(1))
        self.assertIn('in flight', (await self.receive(socket))[0]['error'])
        self.fixture.gate.set()
        await socket.close()
        await self.released()
        self.assertEqual(self.fixture.cancelled, 1)

    async def test_corrupt_backend_output_fails_and_cancels(self):
        for corruption in ('source', 'count', 'geometry', 'pixels', 'boolean'):
            with self.subTest(corruption=corruption):
                self.fixture.corrupt = corruption
                socket = await self.connect()
                await self.start(socket)
                await socket.send_bytes(self.chunk())
                metadata, _ = await self.receive(socket)
                self.assertEqual(metadata['op'], 'error')
                await socket.close()
                await self.released()

    async def test_streaming_prompt_round_trip(self):
        bundle, recipe = self.fixture.artifacts['synthetic-echo']
        recipe = replace(recipe, protocol=FIXED_STREAMING_TEXT_RECIPE)
        self.fixture.artifacts['synthetic-echo'] = (bundle, recipe)
        self.gateway.artifacts['synthetic-echo'] = (bundle, recipe)
        finish = self.fixture.controller.finish
        def with_prompt_receipt(job, pixels, timings):
            if job.op == 'chunk':
                text = job.metadata['prompt']['text']
                timings = dict(applied_prompt=dict(text=text, sha256=hashlib.sha256(text.encode()).hexdigest(),
                                                  first_chunk=0, rows=3))
            return finish(job, pixels, timings)
        self.fixture.controller.finish = with_prompt_receipt
        socket = await self.connect()
        await self.start(socket)
        prompt = dict(revision=1, text='Turn left.', emitted_ns=0)
        await socket.send_bytes(self.chunk(prompt=prompt))
        metadata, _ = await self.receive(socket)
        self.assertEqual(metadata['op'], 'output')
        self.assertEqual(metadata['submitted_prompt'], prompt)
        self.assertEqual(metadata['source_frames'][0]['prompt_revision'], 1)
        await socket.send_bytes(self.chunk(1, True, prompt=prompt))
        metadata, _ = await self.receive(socket)
        self.assertEqual(metadata['status'], 'complete')
        await socket.close()

    async def test_oversized_websocket_terminates_active_session(self):
        socket = await self.connect()
        await self.start(socket)
        await socket.send_bytes(bytes(self.gateway.config['max_input_bytes']+1))
        message = await socket.receive(timeout=3)
        self.assertIn(message.type, (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.BINARY))
        await socket.close()
        await self.released()
        self.assertEqual(self.fixture.cancelled, 1)

    async def test_uncertain_cleanup_quarantines_slot(self):
        socket = await self.connect()
        await self.start(socket)
        self.fixture.cancel_fails = True
        await socket.close()
        async with asyncio.timeout(3):
            while self.gateway.quarantine is None:
                await asyncio.sleep(.01)
        with self.assertRaises(WSServerHandshakeError) as caught:
            await self.connect()
        self.assertEqual(caught.exception.status, 409)
        self.assertIsNotNone(self.gateway.owner)

    async def test_busy_external_backend_does_not_quarantine_gateway(self):
        await asyncio.to_thread(self.gateway.client.call, '/session', dict(session_id='external',
            artifact_id='synthetic-echo', seed=1, condition_seed=1,
            prompt=self.fixture.spec['system_prompt']), bytes(6))
        socket = await self.connect()
        await socket.send_bytes(pack(dict(op='start', seed=2026, condition_seed=2026), bytes(6)))
        metadata, _ = await self.receive(socket)
        self.assertEqual(metadata['op'], 'error')
        await socket.close()
        await self.released()
        self.assertIsNone(self.gateway.quarantine)
        self.assertEqual(self.fixture.controller.active.session_id, 'external')
        self.assertEqual(self.fixture.cancelled, 0)
        await asyncio.to_thread(self.gateway.client.call, '/session/external/cancel', dict(session_id='external'))
        socket = await self.connect()
        await self.start(socket)
        await socket.close()
        await self.released()

    async def test_requires_explicit_source_and_artifact_selection(self):
        for options in ({}, {'backend':'http://127.0.0.1', 'fixture':True}):
            with self.assertRaises(ValueError):
                await create_app(**options)
        fixture = ControlledFixture()
        fixture.health['artifacts']['second'] = fixture.health['artifacts']['synthetic-echo']
        with self.assertRaisesRegex(ValueError, 'multiple artifacts'):
            await create_app(fixture=True, fixture_backend=fixture)


class StandaloneTests(unittest.TestCase):
    def test_public_access_requires_url_and_private_key(self):
        self.assertIsNone(read_public_access(None, None))
        with tempfile.TemporaryDirectory() as directory:
            key_path = Path(directory) / 'access-key'
            key_path.write_text('test-only-access-key-' * 3)
            for value in ('https://gateway.example.com/proxy/instance_1:8765/', 'https://example.com:8443/'):
                access = read_public_access(value, key_path)
                restarted = read_public_access(value, key_path)
                self.assertEqual(access.key, restarted.key)
                self.assertNotEqual(access.session_token, restarted.session_token, 'restart revokes old cookies')
            for value in ('', 'http://example.com/', 'https://example.com', 'https://example.com/proxy',
                          'https://example.com/?x=1', 'https://example.com/#x', 'https://user@example.com/',
                          'https://example.com:0/', 'https://example.com:65536/', 'https://example.com:/',
                          'https://example.com/\n', 'https://example .com/', 'https://example.com/evil\\/',
                          'https://example.com/?', 'https://example.com/#', 'https://example.com/x;y/',
                          'https://example.com/../', 'https://example.com/%2e%2e/', 1, False):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    read_public_access(value, key_path)
            with self.assertRaises(ValueError):
                read_public_access('https://example.com/', None)
            with self.assertRaises(ValueError):
                read_public_access(None, key_path)
            key_path.write_text('too-short')
            with self.assertRaisesRegex(ValueError, '32 to 1024'):
                read_public_access('https://example.com/', key_path)

    def test_public_binding_requires_explicit_proxy(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, '-m', 'tools.serve_web', '--fixture', '--host', '0.0.0.0'],
                                cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('requires --public-url and --access-key-file', result.stderr)

    def test_launcher_cpu_import_and_help(self):
        root = Path(__file__).resolve().parents[1]
        check = subprocess.run([sys.executable, '-c',
            'from tools.serve_web import bootstrap; bootstrap(); '
            'from projects.minimax_h3.inference.web_serving import FixtureBackend; '
            'import sys; FixtureBackend(); assert "torch" not in sys.modules; '
            'assert "common" not in sys.modules'], cwd=root, capture_output=True, text=True)
        self.assertEqual(check.returncode, 0, check.stderr)
        help_result = subprocess.run([sys.executable, '-m', 'tools.serve_web', '--help'],
            cwd=root, capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn('--fixture', help_result.stdout)


if __name__ == '__main__':
    unittest.main()
