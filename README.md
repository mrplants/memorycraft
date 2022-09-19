# memorycraft

Experimentation with the MineRL[^1] dataset.

The MineRL v0.4 dataset is a set of screen captures from Minecraft users accomplishing simple tasks (e.g., find a cave, chop down a tree with an axe, navigate a dense landscape, etc.).  This repository experiments with different AI methods on the MineRL datasets.

## Idea 1

Cluster the frames using kmeans[^2][^3].

This is a baseline clustering that allows me to wrap my arms around this dataset while employing a simple ML technique called kmeans.  Since there are far too many frames to fit in memory, my kmeans implementation uses batched online averaging with centroid updates at the end of each iteration.  I could have used a minibatch update approach, but I wanted guaranteed convergence.  Wherever possible, I used Tensorflow batched operations to speed up the data processing.  Here are the centroids resulting from a small sample of the data (10,000 frames):

<p align="center">
    <img width="461" height="369" src="https://user-images.githubusercontent.com/3487464/190886530-185c7321-088c-4fff-8904-102eaf054f85.png">
</p>

[^1]: Guss, William H., et al. "Minerl: A large-scale dataset of minecraft demonstrations." arXiv preprint arXiv:1907.13440 (2019).

[^2]: J. MacQueen (1967). Some methods for classification and analysis of multivariate observations. Proc. Fifth Berkeley Symp. on Math. Statist. and Prob., Vol. 1 (Univ. of Calif. Press, 1967), 281--297.

[^3]: Steinhaus (1956). Sur la division des corps mat ́eriels en parties. Bulletin de l’Académie Polonaise des Sciences, Classe III, vol. IV, no. 12, 801-804.
