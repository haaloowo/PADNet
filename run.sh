#!/bin/bash
cd /media/yf/CODE/Code/RDAD/myRDAD-complex2-adaptive_memory
source activate diad
python main.py --category carpet
python main.py --category screw
python main.py --category tile
python main.py --category toothbrush
python main.py --category transistor
python main.py --category wood
python main.py --category zipper
python main.py --category metal_nut
python main.py --category leather
python main.py --category bottle
python main.py --category cable
python main.py --category capsule
python main.py --category grid
python main.py --category hazelnut
python main.py --category pill
conda deactivate
echo "end"
exit 0
