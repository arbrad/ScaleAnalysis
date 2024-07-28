Extremely prototype code. While already a rewrite, it suffers hugely from the effects of further research. I will rewrite it before publication.

The main idea of this work is as follows:

1. Use the UI to draw an ideal graph over a given natural mosaic.

2. Analyze the perpendicularity and bisectionality characteristics of the graph, which together are the two properties of Voronoi diagrams. As the generators are unknown, a convex program searches for optimal locations via a convex optimization criterion --- the sum of the normalized perpendicularity and bisectionality metrics across all adjacent polygons.

3. Randomly perturb the vertex locations while preserving convexity. Compare the metrics of a sample set of perturbations to the true metrics to determine statistical significance.

4. Also compare against the centroids. If the metrics for centroids as generators are near the optimal ones and far from the distribution found in 3, conclude that the underlying process is actually centroidal Voronoi.

5. Analyze uniformity metrics to determine if the centroids or discovered optimal generators are blue noise distributed, which would induce centroidality.

6. Given that animals often have axial bias in growth, the analysis includes an analysis that finds and eliminates axial bias. The transformation only affects perpendicularity.

NOTE: Directory data is currently empty because of pre-publication constraints.

Structure: src contains 

 - lib.py, which contains the core concepts (vertices, polygons, tessellations, Voronoi tessellations) and analyses (axial bias, perpendicularity and bisectionality, convex program to find optimal generators, random uniform perturbation that preserves convexity)
  - scale2tess.py, which contains the UI code to construct tessellations and perform the analyses
  - sim.py, which explores a noisy variant of the Schnakenberg diffusion-reaction equation that produces what we're seeing in noisy regions of amniote scales