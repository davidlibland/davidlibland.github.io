---
title: Visualizing Sorts Through Origami
---

In this post, we'll talk about 
[Origami programming](http://www.cs.ox.ac.uk/people/jeremy.gibbons/publications/origami.pdf)
(since it involves a lot of folding and unfolding). As
a fun application, we'll use it to visualize a couple sorting algorithms.

<!-- Haskell Prelude:
```haskell
{-# LANGUAGE UndecidableInstances #-}
import Data.Bifunctor
import Data.Bifoldable
import Data.Bitraversable
import Data.Monoid
```
-->

Origami programming is a generic programming pattern for recursive data types
such as lists and trees:

```haskell
-- Lists:
data List a = Nil | Cons a (List a)
-- Trees:
data Tree a = Leaf a | Node (Tree a) (Tree a)
```

In the origami perspective, the first step is to replace the recursive reference in
the type constructor with an arbitrary reference, `r`:

```haskell
-- Lists:
data List' a r = Nil' | Cons' a r deriving Show
-- Trees:
data Tree' a r = Leaf' a | Node' r r deriving Show
```

Before proceeding, note that whenever the recursive data structure is well-defined, we can recover 
it as the fixed point of this new type constructor. For instance, `List a` is isomorphic to any type, 
`F a`, such that `List' a (F a) = F a`.

Now, let's play with the resulting bivariate type constructor abstractly. We'll denote the abstract
bivariate type constructor by `s a r`. For example, 

```s a r = List' a r```, or
 
```s a r = Tree' a r```.
 
To begin, 
we'll assume that `s` is natural in both types. That is to say, it is a bifunctor:

$$ s : \mathbf{Hask}\times \mathbf{Hask} \to \mathbf{Hask}$$

For instance, `List'` and `Tree'` are recognized as bifunctors as follows:

```haskell
instance Bifunctor List' where
  bimap f g Nil'        = Nil'
  bimap f g (Cons' x r) = Cons' (f x) (g r)

instance Bifunctor Tree' where
  bimap f g (Leaf' x)       = Leaf' (f x)
  bimap f g (Node' rl rr) = Node' (g rl) (g rr)
```

Now fix the first type, `a` for a moment, and consider the category $\mathbf{Alg}(\texttt{s a})$
whose objects are functions $\texttt{s a b} \to \texttt{b}$, and whose morphisms 
between objects $\texttt{s a b} \to \texttt{b}$ and $\texttt{s a c} \to \texttt{c}$ are commutative squares

![](../images/origami/morphism_square.svg)
<!--
$$
\begin{tikzcd}
  \texttt{s a b} \arrow[rr, dashed, "\texttt{second f}"] \arrow[d] & & \texttt{s a c} \arrow[d]  \\
  \texttt{b} \arrow[rr, dashed, "\texttt{f}"] & & \texttt{c}
\end{tikzcd}
$$
-->

Here `second f = bimap id f` applies `f` to the second type-paramter in the bifunctor `s`.


For example, the objects in $\mathbf{Alg}(\texttt{List' a})$ are functions of type
`List' a b -> b`, which (after pattern matching) are given by an element of `b` (which 
`Nil'` is sent to) and a binary function `a -> b -> b` (matched to `Cons' a b`).

Now suppose that $\mathbf{Alg}(\texttt{s a})$ has an [initial object](https://en.wikipedia.org/wiki/Initial_and_terminal_objects)
 $\texttt{s a fix} \xrightarrow{\texttt{Fix}} \texttt{fix}$. We claim that `fix` is a fixed-point of `s a`. To see this, 
 consider the unique morphism:

![](../images/origami/lifted_square_1.svg)
<!--
$$
\begin{tikzcd}
  \texttt{s a fix} \arrow[r, dashed] \arrow[d, "\texttt{Fix}"] & \texttt{s a (s a fix)} \arrow[d, "\texttt{second Fix}"]  \\
  \texttt{fix} \arrow[r, dashed, "\texttt{unFix}"] & \texttt{s a fix}
\end{tikzcd}
$$
-->
where `second Fix = bimap id Fix` is the lift of `Fix` to the functor `s a`. 
We claim that `Fix :: s a fix -> fix` is an isomorphism. To see this, extend the diagram
by adding the arrow `Fix :: s a fix -> fix` at the target of `unFix`:

![](../images/origami/lifted_square_2.svg)
<!--
$$
\begin{tikzcd}
  \texttt{s a fix} \arrow[r, dashed] \arrow[d, "\texttt{Fix}"] \arrow[dr, dotted] & \texttt{s a (s a fix)} \arrow[d, "\texttt{second Fix}"]  \\
  \texttt{fix} \arrow[r, dashed, "\texttt{unFix}"] \arrow[dr, dotted] & \texttt{s a fix} \arrow[d, "\texttt{Fix}"]\\
& \texttt{fix}
\end{tikzcd}
$$
-->

The composition produces a new commutative square:

![](../images/origami/lifted_square_3.svg)
<!--
$$
\begin{tikzcd}
  \texttt{s a fix} \arrow[d, "\texttt{Fix}"] \arrow[r, dotted] & \texttt{s a fix} \arrow[d, "\texttt{Fix}"]  \\
  \texttt{fix} \arrow[ur, dashed, "\texttt{unFix}"'] \arrow[r, dotted] & \texttt{fix} 
\end{tikzcd}
$$
-->

Since `Fix :: s a fix -> fix` is an initial object, the horizontal dotted arrows are uniquely determined by the vertical
arrows, and must therefore both be the identity maps. Hence `unFix . Fix = id` and `Fix . unFix = id`, as claimed.

This shows that $\texttt{s a fix} \cong \texttt{fix}$ is an isomporhism. In other words, `fix` is a fixed point of 
`s a`, which allows us to unwind it recurively as:

$$\texttt{fix} \cong \texttt{s a fix}\cong \texttt{s a (s a fix)}\cong \texttt{s a (s a (s a fix))}\dots$$

For instance, the fixed point of `List' a` can be unwound to

```
Nil' | Cons' a Nil' | Cons' a (Cons' a Nil') | ...
```

that is, either an empty list, a list with one element, a list with two elements, etc.

Ok, so we've shown that the initial object of $\mathbf{Alg}(\texttt{s a})$ is the 
recursive datatype we're interested in, and we may identify it concretely as the fixed point
`Fix s a` using the following construction

```haskell
newtype Fix s a = Fix {unFix :: s a (Fix s a)}
```
<!--
```haskell
-- We can show these (requires UndecidableInstances)
instance Show (s a (Fix s a)) => Show (Fix s a) where
    show x = "(" ++ show (unFix x) ++ ")"
```
-->

We can use `Fix` to define the recursive datatypes, such as lists:
```haskell
-- Now define the recursive datatypes:
type ListF a = Fix List' a
-- along with convenient (lifted) constructors:
nil' = Fix Nil'
infixr 1 #
(#) x y = Fix $ Cons' x y

-- Here's an example list: [1, 2, 3, 4]
aList :: ListF Int
aList = 1 # 2 # 3 # 4 # nil'
```

Or trees:
```haskell
-- Define the recursive datatype, along with lifted constructors
type TreeF a = Fix Tree' a
leaf' = Fix . Leaf'
node' l r = Fix $ Node' l r

-- Here's an example tree: 
aTree :: TreeF Int
aTree = node' (node' (leaf' 2) (leaf' 4)) (node' (leaf' 6) (leaf' 8))
```

Moreover, generic programing allows all our recursive datatypes to inherit various properties.
For instance, the fact that `s` was a bifunctor implies that `Fix s` is a functor:

```haskell
instance Bifunctor s => Functor (Fix s) where
    fmap f x = Fix  $ bimap f (fmap f) (unFix x)
```

Similarly, if `s` is bitraversable, then `Fix s` becomes traversable.
<!--
```haskell
instance Bifoldable s => Foldable (Fix s) where
    foldMap f x = bifoldMap f (foldMap f) (unFix x)

instance Bitraversable s => Traversable (Fix s) where
    traverse f x = Fix <$> bitraverse f (traverse f) (unFix x)
```

```haskell
instance Bifoldable List' where
    bifoldMap f g Nil' = mempty
    bifoldMap f g (Cons' x r) = (f x) <> (g r)

instance Bitraversable List' where
    bitraverse f g Nil' = pure Nil'
    bitraverse f g (Cons' x r) = Cons' <$> (f x) <*> (g r)
```

```haskell
instance Bifoldable Tree' where
    bifoldMap f _ (Leaf' x) = f x
    bifoldMap _ g (Node' l r) = (g r) <> (g r)

instance Bitraversable Tree' where
    bitraverse f _ (Leaf' x) = Leaf' <$> f x
    bitraverse _ g (Node' l r) = Node' <$> (g l) <*> (g r)
```
-->

What else does this perspective get us? We get origami: *folds* and *unfolds* (also known as catamorphisms, and 
anamorphisms, respectively).

### Folds:

Given a function `f :: s a b -> b`, the fact `s a (Fix s a) -> Fix s a` is an initial object implies
that there exists a unique pair of maps:

![](../images/origami/fold_initial_object.svg)
<!--
$$
\begin{tikzcd}
  \texttt{s a (Fix s a)} \arrow[rr, dashed, "\texttt{second (gfold f)}"] \arrow[d, "Fix", shift left=1ex] &&\texttt{s a b} \arrow[d, "f"]  \\
  \texttt{(Fix s a)} \arrow[rr, dashed, "\texttt{gfold f}"] \arrow[u, "unFix", shift left=1ex] && \texttt{b}
\end{tikzcd}
$$
-->

The function `gfold f :: Fix s a -> b` is a generalization of `fold` to all recursive data types;
we can read it's definition directly off the diagram above:

```haskell
gfold :: Bifunctor s => (s a b -> b) -> Fix s a -> b
gfold f = f . second (gfold f) . unFix
```

Let's look at a couple examples. First consider aggregating a monoid over our list:
```haskell
aggList :: Monoid a => List' a a -> a
aggList Nil' = mempty
aggList (Cons' x r) = x <> r
```

This lets us add the values in our list:
```
print $ (gfold aggList) $ fmap Sum aList
>> Sum {getSum = 10}
```
multiply them:
```
print $ (gfold aggList) $ fmap Product aList
>> Product {getProduct = 24}
```
or convert it to a standard list:
```haskell
toList :: ListF a -> [a]
toList = gfold aggList . fmap (\x -> [x])
```
```
print $ toList aList
>> [1,2,3,4]
```

We can play the same game to aggregate monoids over trees:
```haskell
aggTree :: Monoid a => Tree' a a -> a
aggTree (Leaf' x) = x
aggTree (Node' l r) = l <> r
```

This lets us count the leaves in our tree:
```
print $ (gfold aggTree) $ fmap (\_ -> Sum 1) aTree
>> Sum {getSum = 10}
```
convert it to a standard list by a walking from left to right:
```haskell
flatten :: TreeF a -> [a]
flatten = gfold aggTree . fmap (\x -> [x])
```
```
print $ flatten aTree
>> [2,4,6,8]
```
or check for any odd elements:
```
isOdd x = x `mod` 2 /= 0
print $ (gfold aggTree) $ fmap (Any . isOdd) aTree
>> Any {getAny = False}
```

### Unfolds:

We may also play the dual game, and consider the category $\mathbf{CoAlg}(\texttt{s a})$
whose objects are functions $\texttt{b}\to \texttt{s a b}$, and whose morphisms 
between objects $\texttt{b}\to \texttt{s a b}$ and $\texttt{c} \to \texttt{s a c}$ are commutative squares

![](../images/origami/morphism_square_dual.svg)
<!--
$$
\begin{tikzcd}
  \texttt{b} \arrow[rr, dashed, "\texttt{f}"] \arrow[d] & & \texttt{c}\arrow[d] \\
  \texttt{s a b} \arrow[rr, dashed, "\texttt{second f}"] & & \texttt{s a c} 
\end{tikzcd}
$$
-->

By construction[^terminal_assumptions], `unFix :: Fix s a -> s a (Fix s a)` is a [terminal object](https://en.wikipedia.org/wiki/Initial_and_terminal_objects)
for $\mathbf{CoAlg}(\texttt{s a})$. Indeed, for any function `g :: b -> s a b` we may recursively construct the map 

[^terminal_assumptions]: One needs to make a few additional assumptions for this to prove that 
`unFix :: Fix s a -> s a (Fix s a)` is terminal, namely that functions to 
$$\texttt{Fix s a}\cong\texttt{s a (Fix s a)}\cong \texttt{s a (s a (Fix s a))}\cong \cdots$$ 
can be defined recursively.

```
phi :: b -> Fix s a
phi = Fix . second phi . g
```

Since `Fix` is an inverse for `unFix`, this fits into a commutative square:

![](../images/origami/unfold_terminal_obj_phi.svg)
<!--
$$
\begin{tikzcd}
 \texttt{b} \arrow[d, "g"] \arrow[rr, dashed, "\texttt{phi}"]&& \texttt{Fix s a}  \arrow[d, "unFix", shift left=1ex] \\
 \texttt{s a b} \arrow[rr, dashed, "\texttt{second phi}"', shift right=.5ex]  && \texttt{s a (Fix s a)}  \arrow[u, "Fix", shift left=1ex]
\end{tikzcd}
$$
-->

Conversely, any function `phi` fitting into the commutative square above must be given by the same recursive formula[^from_diagram]; 
which proves that `Fix s a` is terminal.

[^from_diagram]: This statement is a tautology, since the diagram and the recursive formula are equivalent -
assuming functions to `Fix s a` can be defined recursively. Said differently, the statement that 
`unFix :: Fix s a -> s a (Fix s a)` is terminal is equivalent to 
```
gunfold :: Bifunctor s => (b -> s a b) -> b -> Fix s a
gunfold g = Fix . second (gunfold g) . g
```
being a well defined map.

Of course, `phi` depends on `g :: b -> s a b`, and it makes sense to rewrite it as such:

```haskell
gunfold :: Bifunctor s => (b -> s a b) -> b -> Fix s a
gunfold g = Fix . second (gunfold g) . g
```

<!--
![](../images/origami/unfold_terminal_object.svg)
$$
\begin{tikzcd}
 \texttt{b} \arrow[d, "g"] \arrow[rr, dashed, "\texttt{gunfold g}"]&& \texttt{Fix s a}  \arrow[d, "unFix", shift left=1ex] \\
 \texttt{s a b} \arrow[rr, dashed, "\texttt{second (gunfold g)}"', shift right=.5ex]  && \texttt{s a (Fix s a)}  \arrow[u, "Fix", shift left=1ex]
\end{tikzcd}
$$
-->

The function `gunfold g :: b -> Fix s a` allows us to *unfold* an instance of `b` into an instance of the recursive 
datatype `Fix s a`.

Let's look at a couple examples:

As a simple example, consider decomposing an integer into its digits (in ascending order):
```haskell
-- Our iterative step just peels off the smallest digit:
onesAndTens :: Int -> List' Int Int
onesAndTens x = if x <= 0 then Nil' else Cons' (x `mod` 10) (x `div` 10)

-- The full function just iterates:
digits = gunfold onesAndTens
```

```
digits 52341
>> 1 # 4 # 3 # 2 # 5 # nil'
```

In general, unfolds for `List'` are "generators": they are described by a state type `b`, together
with a map `next :: b -> List' a b` which returns either `Nil'` (terminate the list), or the pair `Cons' a b` 
(produce the next list element of type `a` along with the next state of type `b`). The function
`gunfold next :: b -> ListF a` iteratively generates the corresponding list given an initial state. 

For instance to (inefficiently) list the prime numbers, we can use 
[Eratosthene's sieve](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes), where `b = [Int]` describes the state 
of the sieve. At each stage, we peel off the smallest element of the sieve (which is prime), and update the
sieve by filtering for those elements which are coprime to that prime:
```haskell
nextPrime :: [Int] -> List' Int [Int]
nextPrime (prime:sieve) = Cons' prime (filter (coPrimeTo prime) sieve) where
  coPrimeTo n = (/= 0) . mod n

primes = gunfold nextPrime [2..] 
```
````
primes
>> 2 # 3 # 5 # 7 # 11 # 13 # 17 # ...
````

We can also *unfold* values into trees. For instance, suppose we want to factorize
an integer. Let 
```haskell
factorPair :: Integer -> Maybe (Integer, Integer)
```
<!--
```haskell
maybeHead :: [a] -> Maybe a
maybeHead (x:xs) = Just x
maybeHead _ = Nothing

isSquare :: (Integral a) => a -> Bool
isSquare n = (round . sqrt $ fromIntegral n) ^ 2 == n

squareRoot :: Integer -> Integer
squareRoot = floor . sqrt . (fromIntegral :: Integer -> Double)

factorPair 1 = Nothing
factorPair 2 = Nothing
factorPair x = if x `mod` 2 == 0 then Just (2, x `div` 2) else maybeHead $ 
    filter ((>1) . fst) $
    map (\(a2, b) -> let a = squareRoot a2 in (a-b, a+b)) $
    filter (isSquare . fst) 
    [(x+b*b, b) | b<-[0,1.. squareRoot x]]
```
-->
Be a function which decomposes a composite integer into Just a pair of factors or Nothing for a prime.
Then we can *unfold* an integer into a factor tree:

```haskell
factorTree = gunfold f where
    f b = case factorPair b of
        Nothing -> Leaf' b
        (Just (x, y)) -> Node' x y
```
```
print $ factorTree 60
>> Node' (Leaf' 2) (Node' (Leaf' 2) (Node' (Leaf' 3) (Leaf' 5)))
```

### Visualizing Sorts

Many sorting algorithms involve a computational tree, which we can explicitly instantiate and visualize 
by decomposing the algorithms as 

$$\texttt{List} \xrightarrow{\texttt{gunfold}} \texttt{Tree} \xrightarrow{\texttt{gfold}} \texttt{List}$$

#### Merge Sort
Consider [merge sort](https://en.wikipedia.org/wiki/Merge_sort). Starting with the tree structure:

```haskell
data MTree' a r = MEmpty | MLeaf a | MNode r r deriving Show
type MTreeF a = Fix MTree' a

-- Describe the bifunctor structure:
instance Bifunctor MTree' where
  bimap _ _ MEmpty = MEmpty
  bimap f _ (MLeaf x) = MLeaf (f x)
  bimap _ g (MNode l r) = MNode (g l) (g r)
```

Merge sort starts by successively splitting a list:
```haskell
split :: [a] -> MTree' a [a]
split [] = MEmpty
split [x] = MLeaf x
split xs = MNode (take n xs) (drop n xs) where n = length xs `div` 2
```

and then successively merging the pieces together in ascending order:
```haskell
merge :: Ord a => MTree' a [a] -> [a]
merge MEmpty = []
merge (MLeaf x) = [x]
merge (MNode xs []) = xs
merge (MNode [] ys) = ys
merge (MNode (x:xs) (y:ys)) = if x <= y then x:merge (MNode xs (y:ys)) else y:merge (MNode (x:xs) ys)
```

The full merge sort is just the composite:
```haskell
mSort :: Ord a => [a] -> [a]
mSort = gfold merge . gunfold split
```
```
print $ mSort [3,5,2,1,-4,6]
>> [-4,1,2,3,5,6]
```

Using the [GraphViz package](https://hackage.haskell.org/package/graphviz-2999.20.0.4/docs/Data-GraphViz.html), 
one may programatically convert the intermediate `QTreeF`'s into graphs, allowing us to visualize the computation:

![](../images/origami/mTree.png)

#### Quick Sort

[Quick sort](https://en.wikipedia.org/wiki/Quicksort) also involves a computational tree: at each stage
one 

1. arbitrarily chooses an element of the list on which to *pivot*, and
2. moves all elements smaller than the pivot the the left of the pivot,
3. moves all elements larger than the pivot to the right of the pivot.
4. Repeats this process on both the sublist to the left of the pivot and the sublist to the right of the pivot.

The computation involves the following tree structure:
```haskell
data QTree' a r = QEmpty | QLeaf a | QNode a r r
type QTreeF a = Fix QTree' a

-- Describe the bifunctor structure:
instance Bifunctor QTree' where
  bimap _ _ QEmpty = QEmpty
  bimap f _ (QLeaf x) = QLeaf (f x)
  bimap f g (QNode x l r) = QNode (f x) (g l) (g r)

-- A custom show method:
instance (Show a, Show r) => Show (QTree' a r) where
    show (QLeaf x) = show x
    show (QNode x l r) = show l ++ " pivot: " ++ show x ++ " " ++ show r 
    show _ = ""
```

Pivoting (about the first element) can be described as follows:
```haskell
pivot :: Ord a => [a] -> QTree' a [a]
pivot [] = QEmpty
pivot [x] = QLeaf x
pivot (x:xs) = QNode x (filter (<x) xs) (filter (>=x) xs)
```

After unfolding a list via `pivot`, the leaves are ordered:
```
print $ gunfold pivot [3,5,2,1,-4,6]
>> ((((-4) pivot: 1 ()) pivot: 2 ()) pivot: 3 (() pivot: 5 (6)))
```

So we have nothing left to do except walk the tree from left to right:
```haskell
walk :: QTree' a [a] -> [a]
walk (QLeaf x) = [x]
walk (QNode x l r) = l ++ x:r
walk _ = []
```

The full quick-sort is just the composite:
```haskell
qSort :: Ord a => [a] -> [a]
qSort = gfold walk . gunfold pivot
```
```
print $ qSort [3,5,2,1,-4,6]
>> [-4,1,2,3,5,6]
```

Once again, we can programatically visualize the computational tree using the 
[GraphViz package](https://hackage.haskell.org/package/graphviz-2999.20.0.4/docs/Data-GraphViz.html) package:

![](../images/origami/qTree.png)
