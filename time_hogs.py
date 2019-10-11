import sys
import pstats

in_file = sys.argv[1]
out_file = sys.argv[2]

with open(out_file, 'w') as f:
    p = pstats.Stats(in_file, stream=f)
    p.strip_dirs().sort_stats('time').print_stats()

if len(sys.argv) > 3:
    out_file = sys.argv[3]
    with open(out_file, 'w') as f:
        p = pstats.Stats(in_file, stream=f)
        p.strip_dirs().sort_stats('cumulative').print_stats()

