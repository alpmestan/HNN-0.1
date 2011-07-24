-- | Net module, defining functions to work on a neural network, which is a
--   list of list of neurons 
module AI.HNN.Net where

import AI.HNN.Layer
import AI.HNN.Neuron
import Control.Arrow
import Data.List as L
import Data.Vector as V

check :: [[Neuron]] -> Bool
check nss = let l = Prelude.length nss in l > 1 && l < 3

nn :: [[Neuron]] -> [[Neuron]]
nn nss | check nss = nss
       | otherwise = error "Invalid nn"

-- * Computation

-- | Computes the output of the given neural net on the given inputs
computeNetU :: [[Neuron]] -> Vector Double -> Vector Double
computeNetU neuralss xs = 
  computeLayerU (nss !! 1) $ computeLayerU (Prelude.head nss) xs
  where nss = nn neuralss 
  
-- | Computes the output of the given neural net on the given inputs
computeNet :: [[Neuron]] -> [Double] -> [Double]
computeNet neuralss = toList . computeNetU neuralss . fromList

-- * Quadratic Error

-- | Returns the quadratic error of the neural network on the given sample
quadErrorNetU :: [[Neuron]] -> (Vector Double, Vector Double) -> Double
quadErrorNetU nss (xs,ys) = (/2.0) $ 
  V.sum . V.zipWith (\y s -> (y - s)**2) ys $ 
    computeNetU nss xs

-- | Returns the quadratic error of the neural network on the given sample
quadErrorNet :: [[Neuron]] -> ([Double], [Double]) -> Double
quadErrorNet nss = quadErrorNetU nss . (fromList *** fromList)

-- | Returns the quadratic error of the neural network on the given samples
globalQuadErrorNetU :: [[Neuron]] -> [(Vector Double, Vector Double)] -> Double
globalQuadErrorNetU nss = L.sum . L.map (quadErrorNetU nss)

-- | Returns the quadratic error of the neural network on the given samples
globalQuadErrorNet :: [[Neuron]] -> [([Double], [Double])] -> Double
globalQuadErrorNet nss = globalQuadErrorNetU nss . L.map (fromList *** fromList)

-- * Learning

-- | Train the given neural network using the backpropagation algorithm on the
--   given sample with the given learning ratio (alpha)
backPropU :: Double -> [[Neuron]] -> (Vector Double, Vector Double) -> [[Neuron]]
backPropU alpha nss (xs, ys) = [aux (L.head nss) ds_hidden xs
                        ,aux (nss !! 1) ds_out output_hidden]
    where 
      output_hidden = computeLayerU (L.head nss) xs
      output_out = computeLayerU (nss !! 1) output_hidden
      ds_out = V.zipWith (\s y -> s * (1 - s) * (y - s)) output_out ys
      ds_hidden = V.zipWith (\x s -> x * (1-x) * s) output_hidden
                . fromList $ 
                    L.map (V.sum . V.zipWith (*) ds_out)
                  . L.map fromList 
                  . transpose 
                  . L.map (toList . weights) $ 
                      (nss !! 1)
      aux ns ds xs = L.zipWith (\n d -> n { weights = V.zipWith (\w x -> w + alpha * d * x) (weights n) xs }) ns (toList ds)

-- | Train the given neural network using the backpropagation algorithm on the
--   given sample with the given learning ratio (alpha)
backProp :: Double -> [[Neuron]] -> ([Double], [Double]) -> [[Neuron]]
backProp alpha nss = backPropU alpha nss . (fromList *** fromList)

trainAux :: Double -> [[Neuron]] -> [(Vector Double, Vector Double)] -> [[Neuron]]
trainAux alpha = L.foldl' (backPropU alpha)

-- | Train the given neural network on the given samples using the
--   backpropagation algorithm using the given learning ratio (alpha) and the
--   given desired maximal bound for the global quadratic error on the samples
--   (epsilon)
trainU :: Double -> Double -> [[Neuron]] -> [(Vector Double, Vector Double)] -> [[Neuron]]
trainU alpha epsilon nss samples = until 
  (\nss' -> globalQuadErrorNetU nss' samples < epsilon) 
  (\nss' -> trainAux alpha nss' samples) 
  nss

-- | Train the given neural network on the given samples using the
--   backpropagation algorithm using the given learning ratio (alpha) and the
--   given desired maximal bound for the global quadratic error on the samples
--   (epsilon)
train :: Double -> Double -> [[Neuron]] -> [([Double], [Double])] -> [[Neuron]]
train alpha epsilon nss = trainU alpha epsilon nss . L.map (fromList *** fromList)
