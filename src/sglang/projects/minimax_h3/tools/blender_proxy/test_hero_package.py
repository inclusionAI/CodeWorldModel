import unittest

import numpy as np

from tools.blender_proxy.hero_package import STAGES, atlas_layout, make_atlas


class HeroAtlasTests(unittest.TestCase):
    def test_each_tile_keeps_its_orientation_and_repeats_its_own_edges(self):
        layout = atlas_layout(16, 9)
        images = {}
        for index,stage in enumerate(STAGES):
            pixels = np.zeros((9,16,3), dtype=np.uint8)
            pixels[:,:,0],pixels[:,:,1],pixels[:,:,2] = index*60,np.arange(16),np.arange(9)[:,None]
            images[stage] = pixels
        atlas = make_atlas(images,layout)
        for stage,(x,y,width,height) in layout['rects'].items():
            np.testing.assert_array_equal(atlas[y:y+height,x:x+width],images[stage])
            np.testing.assert_array_equal(atlas[y-16:y,x-16:x+width+16],np.pad(images[stage][:1],((15,0),(16,16),(0,0)),mode='edge'))
            for dx,dy,sx,sy in [(-16,-16,0,0),(width+15,-16,-1,0),(-16,height+15,0,-1),(width+15,height+15,-1,-1)]:
                np.testing.assert_array_equal(atlas[y+dy,x+dx],images[stage][sy,sx])
        self.assertEqual(atlas_layout(640,360)['width'],1344)
        self.assertEqual(atlas_layout(640,360)['height'],784)

    def test_wrong_source_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Wrong source size'):
            make_atlas({'rgb':np.zeros((4,4,3),dtype=np.uint8)},atlas_layout(16,9))


if __name__ == '__main__':
    unittest.main()
