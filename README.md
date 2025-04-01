# NeuRep_GWOT Analysis Pipeline

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