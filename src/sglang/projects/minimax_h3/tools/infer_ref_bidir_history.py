"""Stream a proxy bundle with a bidirectional recent-history Ref artifact."""
from projects.minimax_h3.modeling.ref_recipe import BIDIR_RECIPE
from projects.minimax_h3.tools.infer_ref_fixed_history import _main


def main(argv=None):
    return _main(argv, expected_protocol=BIDIR_RECIPE)


if __name__ == '__main__':
    main()
