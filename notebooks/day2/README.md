<img src="../../images/supercomputer.png" alt="Parallel MODFLOW Course" style="width:50;height:20">

# Day 2

This directory contains a series of notebooks to create structured and unstructured MODFLOW models that can be built from raw data. The notebooks can be used to build the hypothetical watershed model from [Hughes et al.(2023)](https://doi.org/10.1111/gwat.13327) using a structured and unstructured grid and a model similar to the hypothetical box model from [Verkaik et al. (2021)](https://doi.org/10.1016/j.envsoft.2021.105092) using a structured grid. 

## Step 1

The `step1_build_basin_structured`, `step1_build_basin_unstructured`, or `step1_build_box_structured` notebooks can be used to build the base models for the structured and unstructured hypothetical watershed models and the structured box model.

## Step 2

The `setp2_split_base_model` notebook can be used to split the structured and unstructured hypothetical watershed models and the structured box model using the FloPy model splitter. Structured models can be split into a regular number of rows and columns or using METIS. The unstructured hypothetical watershed model is split using METIS>

## Step 3

The `step3_compare_results` notebook can be used to compare base model and split model results for the structured and unstructured hypothetical watershed models and the structured box model.

## Step 4

The `step4_process_performance` notebook can be used to process the base model and split model performance data written to the simulation  listing files `mfsim*.lst` for the structured and unstructured hypothetical watershed models and the structured box models.

## Step 5

The `step5_evaluate_performance` notebook can be used to plot the base model and split model performance data for the structured and unstructured hypothetical watershed models and the structured box models.

## Miscellaneous Notebooks

* `update_petsc_solver` notebook can be used to modify the PETSc solver settings for the structured and unstructured hypothetical watershed models and the structured box models.



