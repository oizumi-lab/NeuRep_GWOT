# NeuRep_GWOT Analysis Pipeline

This is the analysis code for ["Unsupervised alignment reveals structural commonalities and differences in neural representations of natural scenes across individuals and brain areas"](https://www.cell.com/iscience/fulltext/S2589-0042(25)00688-1).

This repository is for the analysis of human fMRI data. For the analysis code of mouse neuropixels data, please refer to [this repository](https://github.com/oizumi-lab/NeuRep_GWOT_mouse).

## Preprocessing
1. **Download NSD Data**  
   Follow the instructions provided [here](https://cvnlab.slite.page/p/dC~rBTjqjb/How-to-get-the-data) to download the NSD dataset.

2. **Extract Stimulus Labels**  
   Use `preprocess_labels.py` to retrieve label information for the stimuli.

3. **Compute RDMs**  
   Run `preprocess_rdms.py` to calculate the Representational Dissimilarity Matrices (RDMs) for each brain area in participant groups.

## Main Analysis
4. **Compute GWOT between the same brain areas**  
   Use `alignment.py` to calculate the GWOT (Gromov-Wasserstein Optimal Transport) between the same brain areas in different participants groups.

5. **Compute GWOT between different brain areas**  
   Run `alignment_across_roi.py` to calculate the GWOT between different brain areas.

## Visualization
6. **Plot GWOT Results**  
   Use `plot_results.py` to visualize the results of the GWOT analysis.

7. **Visualize Enlarged OT Plan**  
   Run `show_enlarged_OT.py` to generate enlarged visualizations of the optimal transportation plan.