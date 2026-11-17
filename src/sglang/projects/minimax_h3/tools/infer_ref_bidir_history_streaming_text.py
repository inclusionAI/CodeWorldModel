"""Stream a proxy bundle with the bidir-history complete-text inference protocol."""
from projects.minimax_h3.modeling.ref_recipe import BIDIR_STREAMING_TEXT_RECIPE
from projects.minimax_h3.tools.infer_ref_fixed_history import _main


def main(argv=None):
    return _main(argv, expected_protocol=BIDIR_STREAMING_TEXT_RECIPE)


if __name__ == '__main__':
    main()
