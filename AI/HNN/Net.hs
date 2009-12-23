-- | Net module, defining functions to work on a neural network, which is a list of list of neurons 
module AI.HNN.Net where

import AI.HNN.Layer
import AI.HNN.Neuron
import Control.Arrow
import Data.List
import Data.Array.Vector

check :: [[Neuron]] -> Bool
check nss = let l = length nss in l > 1 && l < 3

nn :: [[Neuron]] -> [[Neuron]]
nn nss | check nss = nss
       | otherwise = error "Invalid nn"

-- * Computation

-- | Computes the output of the given neural net on the given inputs
computeNetU :: [[Neuron]] -> UArr Double -> UArr Double
computeNetU neuralss xs = let nss = nn neuralss in computeLayerU (nss !! 1) $ computeLayerU (head nss) xs
  
-- | Computes the output of the given neural net on the given inputs
computeNet :: [[Neuron]] -> [Double] -> [Double]
computeNet neuralss = fromU . computeNetU neuralss . toU

-- * Quadratic Error

-- | Returns the quadratic error of the neural network on the given sample
quadErrorNetU :: [[Neuron]] -> (UArr Double, UArr Double) -> Double
quadErrorNetU nss (xs,ys) = (sumU . zipWithU (\y s -> (y - s)**2) ys $ computeNetU nss xs)/2.0

-- | Returns the quadratic error of the neural network on the given sample
quadErrorNet :: [[Neuron]] -> ([Double], [Double]) -> Double
quadErrorNet nss = quadErrorNetU nss . (toU *** toU)

-- | Returns the quadratic error of the neural network on the given samples
globalQuadErrorNetU :: [[Neuron]] -> [(UArr Double, UArr Double)] -> Double
globalQuadErrorNetU nss = sum . map (quadErrorNetU nss)

-- | Returns the quadratic error of the neural network on the given samples
globalQuadErrorNet :: [[Neuron]] -> [([Double], [Double])] -> Double
globalQuadErrorNet nss = globalQuadErrorNetU nss . map (toU *** toU)

-- * Learning

-- | Train the given neural network using the backpropagation algorithm on the given sample with the given learning ratio (alpha)
backPropU :: Double -> [[Neuron]] -> (UArr Double, UArr Double) -> [[Neuron]]
backPropU alpha nss (xs, ys) = [aux (head nss) ds_hidden xs
                        ,aux (nss !! 1) ds_out output_hidden]
    where 
      output_hidden = computeLayerU (head nss) xs
      output_out = computeLayerU (nss !! 1) output_hidden
      ds_out = zipWithU (\s y -> s * (1 - s) * (y - s)) output_out ys
      ds_hidden = zipWithU (\x s -> x * (1-x) * s) output_hidden . toU $ map (sumU . zipWithU (*) ds_out) . map toU . transpose . map (fromU . weights) $ (nss !! 1)
      aux ns ds xs = zipWith (\n d -> n { weights = zipWithU (\w x -> w + alpha * d * x) (weights n) xs }) ns (fromU ds)

-- | Train the given neural network using the backpropagation algorithm on the given sample with the given learning ratio (alpha)
backProp :: Double -> [[Neuron]] -> ([Double], [Double]) -> [[Neuron]]
backProp alpha nss = backPropU alpha nss . (toU *** toU)

trainAux :: Double -> [[Neuron]] -> [(UArr Double, UArr Double)] -> [[Neuron]]
trainAux alpha = foldl' (backPropU alpha)

-- | Train the given neural network on the given samples using the backpropagation algorithm using the given learning ratio (alpha) and the given desired maximal bound for the global quadratic error on the samples (epsilon)
trainU :: Double -> Double -> [[Neuron]] -> [(UArr Double, UArr Double)] -> [[Neuron]]
trainU alpha epsilon nss samples = until (\nss' -> globalQuadErrorNetU nss' samples < epsilon) (\nss' -> trainAux alpha nss' samples) nss

-- | Train the given neural network on the given samples using the backpropagation algorithm using the given learning ratio (alpha) and the given desired maximal bound for the global quadratic error on the samples (epsilon)
train :: Double -> Double -> [[Neuron]] -> [([Double], [Double])] -> [[Neuron]]
train alpha epsilon nss = trainU alpha epsilon nss . map (toU *** toU)