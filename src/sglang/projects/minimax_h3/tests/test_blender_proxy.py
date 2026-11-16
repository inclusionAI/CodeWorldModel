import tempfile
import unittest
from pathlib import Path

import av
import numpy as np
from PIL import Image

from tools.blender_proxy.package import video


class BlenderProxyVideoTests(unittest.TestCase):
    def test_lossless_video_round_trips_source_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = []
            for index in range(2):
                rgb = np.zeros((384, 672, 3), dtype=np.uint8)
                rgb[:, :] = (60 + index, 170, 240)
                path = root/f'{index:04}.png'
                Image.fromarray(rgb).save(path)
                images.append(path)
            target = root/'proxy.mkv'
            video(images, target, 'ffv1', 'bgr0')
            with av.open(str(target)) as container:
                self.assertEqual(float(container.streams.video[0].average_rate), 24)
                decoded = [f.to_ndarray(format='rgb24') for f in container.decode(video=0)]
            self.assertEqual(len(decoded), len(images))
            for frame, path in zip(decoded, images):
                np.testing.assert_array_equal(frame, np.asarray(Image.open(path).convert('RGB')))


if __name__ == '__main__':
    unittest.main()
