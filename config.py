config = {
    "batch_size": 128,
    "learning_rate": 1e-3,
    "min_learning_rate": 1e-5,
    "optimizer": "ADAM",
    "max_epoch": 200,
    "decay_epoch": 100,
    "decay_cycle": 4,

    "movies_path": "ml-1m/movies.dat",
    "ratings_path": "ml-1m/ratings.dat",
    "users_path": "ml-1m/users.dat",

    "loss": "bpr",  # top_1, cross_entropy
    "embedding": True,  # True when using embedding layer

    "numpy_seed": 10,
    "split_ratio": 0.8,
    "hidden_dim": 40,              # hidden layer dimension of embedding layer
    "sequence_length": 20,          # sequence count of input
    "attention_layer_count": 2,  # count of attention layer
    "negative_sample_count": 100   # count of negative sample for each user
}
