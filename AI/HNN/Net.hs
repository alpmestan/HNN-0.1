-- | Net module, defining functions to work on a neural network, which is a list of list of neurons 
module AI.HNN.Net where

import AI.HNN.Layer
import AI.HNN.Neuron
import Data.List
import Data.Array.Vector

check :: [[Neuron]] -> Bool
check nss = let l = length nss in l > 1 && l < 3

nn :: [[Neuron]] -> [[Neuron]]
nn nss | check nss = nss
       | otherwise = error "Invalid nn"

-- | Computes the output of the given neural net on the given inputs
computeNet :: [[Neuron]] -> UArr Double -> UArr Double
computeNet neuralss xs = let nss = nn neuralss in computeLayer (nss !! 1) $ computeLayer (head nss) xs
                 
-- | Returns the quadratic error of the neural network on the given sample
quadErrorNet :: [[Neuron]] -> (UArr Double, UArr Double) -> Double
quadErrorNet nss (xs,ys) = (sumU . zipWithU (\y s -> (y - s)**2) ys $ computeNet nss xs)/2.0

-- | Returns the quadratic error of the neural network on the given samples
globalQuadErrorNet :: [[Neuron]] -> [(UArr Double, UArr Double)] -> Double
globalQuadErrorNet nss = sum . map (quadErrorNet nss)

-- | Train the given neural network using the backpropagation algorithm on the given sample
backProp :: [[Neuron]] -> (UArr Double, UArr Double) -> [[Neuron]]
backProp nss (xs, ys) = [aux (head nss) ds_hidden xs
                        ,aux (nss !! 1) ds_out output_hidden]
    where 
      output_hidden = computeLayer (head nss) xs
      output_out = computeLayer (nss !! 1) output_hidden
      ds_out = zipWithU (\s y -> s * (1 - s) * (y - s)) output_out ys
      -- ds_hidden = zipWithU (\x s -> x * (1-x) * s) output_hidden $ toU (map (sumU . zipWithU (*) ds_out . weights) (nss !! 1))
      ds_hidden = zipWithU (\x s -> x * (1-x) * s) output_hidden . toU $ map (sumU . zipWithU (*) ds_out) . map toU . transpose . map (fromU . weights) $ (nss !! 1)
      aux ns ds xs = zipWith (\n d -> n { weights = zipWithU (\w x -> w + alpha * d * x) (weights n) xs }) ns (fromU ds)

trainAux :: [[Neuron]] -> [(UArr Double, UArr Double)] -> [[Neuron]]
trainAux = foldl' backProp

-- | Train the given neural network on the given samples using the backpropagation algorithm
train :: [[Neuron]] -> [(UArr Double, UArr Double)] -> [[Neuron]]
train nss samples = until (\nss' -> globalQuadErrorNet nss' samples < (0.1 :: Double)) (\nss' -> trainAux nss' samples) nss
