# Dataset construction

The dataset is distributed separately from this code repository. The published
population is 100,000 random designs plus 117,999 repair proposals, with the
released evaluation statuses and parent-based split retained.

The following existing entry points are included:

| Step | Entry point |
| --- | --- |
| Random design generation | `Generate_V1p1_Design_Batch.py` |
| Evaluate one input shard | `Run_V1p1_System_Evaluation_Batch.py` |
| Parallel batch evaluation | `Run_ShipSystems_Eval_Parallel_v1p1.py` |
| Compile random evaluations | `Compile_V1p1_Dataset.py` |
| Compile and select repair candidates | `Compile_V1p1_Dataset_And_Select_Candidates.py` |
| Run deterministic repair | `python -m parametric_repair_tools.run_repair_pipeline` |
| Parallel below-median repair | `python -m parametric_repair_tools.run_below_median_parallel` |
| Compile repaired results | `Compile_V1p1_Repaired_Dataset.py` and `Compile_V1p1_Repaired_As_Dataset.py` |
| Assemble the unified training release | `Build_Unified_V1p1_Training_Data.py` |
| Build constraint/subsystem mappings | `Build_Constraint_Subsystem_Mapping.py` |

Run each entry point with `--help` for its existing batch identifiers, input
paths, and storage options. Generation, evaluation, repair, and compilation are
separate operations. These tools preserve the existing file conventions; they
do not constitute a one-command reconstruction of every historical batch.

The model workflow consumes the released `Training_Rows.csv.gz`,
`Granular_20_Subsystem_Labels.csv.gz`, and `Shared_Split_Rows.csv.gz` beneath
`Ship_Systems_Designs_v1p1/Unified_Training_Dataset/`. RFR preparation also reads
the performance columns in the same released training table. Preserve those
tables and their source-parent split together.
