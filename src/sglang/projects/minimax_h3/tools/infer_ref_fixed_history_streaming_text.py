"""Stream a proxy bundle with the fixed-history complete-text inference protocol."""
from projects.minimax_h3.modeling.ref_recipe import FIXED_STREAMING_TEXT_RECIPE
from projects.minimax_h3.tools.infer_ref_fixed_history import _main


def main(argv=None):
    return _main(argv, expected_protocol=FIXED_STREAMING_TEXT_RECIPE)


if __name__ == '__main__':
    main()
