# Newton_Shulz
Repo for code for research into the Newton Shulz algorithm

## Generate random orthogonal matrix
Creating a random orthogonal matrix, distributed according to the Haar measure (the cannonical distribution for random orthogonal matrices) can be done using the function defined in [Haar.py](Haar.py). To call the function, use the syntax 
```
G = Haar.randOrth(n)
```
where `n` is the dimension of the orthogonal matrix to be obtained. This will return an n by n numpy array.
