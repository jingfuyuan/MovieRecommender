# Movie recommender system for the MovieLen 20M dataset

The MovieLens 20M dataset contains over 20M ratings. Each row is a rating of a movie given by a user. The users usually only watched a very small portion of all the movies. A recommender system is to recommend not watched movies in this case to a specific user so that the user will very likely rate high scores for the recommended movies.   

I used this project to learn different algoriths and platforms for recommender systems, such as KNN-based collaborative filtering, maxtrix factorization, and neural networks for collaborative filtering.  

Read the Jupyter Notebooks for details

## What I did
- implemented KNN-based collaborative filtering algorith from scratch. I used both user-based and item-based approaches
- implemented maxtrix factorization algorith
- built neural networks with TensorFlow to do collaborative filtering
- did maxtrix factorization with Spark on AWS EMR
- build neural networks with PyTorch, trained and deployed the model on SageMaker to do movie recommendation
- used Graph Neural Networks (GNN) to solve the movie recommendation problem. I build the GNNs with PyG (Pytorch Geometric)