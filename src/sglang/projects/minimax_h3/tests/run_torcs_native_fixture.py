"""CPU-only browser fixture for TORCS -> gateway -> native Omni media adapter.

The native endpoint is FakeOmni, not a model. Run from the repository root.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aiohttp import web
from test_omni_h3_web import FakeOmni
from projects.minimax_h3.inference.omni_h3_web import create_app as create_adapter
from projects.minimax_h3.inference.web_serving import create_app as create_gateway


async def main(port):
    runners = []
    async def serve(app, listen=0):
        runner = web.AppRunner(app, handler_cancellation=True)
        await runner.setup()
        await web.TCPSite(runner, '127.0.0.1', listen).start()
        runners.append(runner)
        return 'http://127.0.0.1:' + str(runner.addresses[0][1])
    try:
        fake = FakeOmni()
        native = web.Application()
        native.router.add_get('/v1/session', fake.handle)
        uri = (await serve(native)).replace('http:', 'ws:') + '/v1/session'
        root = Path(__file__).resolve().parents[1]
        recipe = json.loads((root / 'configs/recipes/omni-h3-longlive.json').read_text())
        artifact = dict(recipe=recipe, encoder_recipe=dict(width=16, height=16,
            proxy_width=16, proxy_height=16, system_prompt='CPU FakeOmni protocol fixture, no model inference.'))
        backend = await serve(create_adapter(uri, 'torcs-native-fixture', artifact, timeout=30))
        url = await serve(await create_gateway(backend=backend, incremental_proxy=True, threejs_racing=True), port)
        print('CPU FakeOmni fixture, NO model inference: ' + url, flush=True)
        await asyncio.Event().wait()
    finally:
        for runner in reversed(runners):
            await runner.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18864)
    asyncio.run(main(parser.parse_args().port))
