#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --error="/global/cfs/cdirs/metatlas/projects/untargeted_tasks/20230223_JGI_MC_508469_AlgaHeatChlUWO241_final_EXP120B_HILICZ_USHXG02066_positive/20230223_JGI_MC_508469_AlgaHeatChlUWO241_final_EXP120B_HILICZ_USHXG02066_positive-mzmine.err"
#SBATCH --output="/global/cfs/cdirs/metatlas/projects/untargeted_tasks/20230223_JGI_MC_508469_AlgaHeatChlUWO241_final_EXP120B_HILICZ_USHXG02066_positive/20230223_JGI_MC_508469_AlgaHeatChlUWO241_final_EXP120B_HILICZ_USHXG02066_positive-mzmine.out"
#SBATCH -A m2650
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH -t 3:00:00

/global/common/software/m2650/mzmine_parameters/MZmine/MZmine-3.7.2/bin/MZmine -threads auto -t /tmp -batch /global/cfs/cdirs/metatlas/projects/untargeted_tasks/20230223_JGI_MC_508469_AlgaHeatChlUWO241_final_EXP120B_HILICZ_USHXG02066_positive/20230223_JGI_MC_508469_AlgaHeatChlUWO241_final_EXP120B_HILICZ_USHXG02066_positive_batch-params.xml -input /global/cfs/cdirs/metatlas/projects/untargeted_tasks/20230223_JGI_MC_508469_AlgaHeatChlUWO241_final_EXP120B_HILICZ_USHXG02066_positive/20230223_JGI_MC_508469_AlgaHeatChlUWO241_final_EXP120B_HILICZ_USHXG02066_positive_filelist.txt
echo MZMINE IS DONE



