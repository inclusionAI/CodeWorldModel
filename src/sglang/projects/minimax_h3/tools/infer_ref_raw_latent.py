"""Run any explicit v2 latent-window artifact using the shared file engine."""
from projects.minimax_h3.tools.infer_ref_fixed_history import _main


def main(argv=None):
    return _main(argv, expected_protocol=None)


if __name__ == '__main__':
    main()
