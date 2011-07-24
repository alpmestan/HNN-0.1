-- | Layer module, defining functions to work on a neural network layer, which
--   is a list of neurons
module AI.HNN.Layer where

import AI.HNN.Neuron
import Control.Arrow
import Data.Vector (Vector(), fromList, sum, toList, zipWith)
import Data.List (foldl')

-- * Layer creation

-- | Creates a layer compound of n neurons with the Sigmoid transfer function,
--   all having the given threshold and weights.
createSigmoidLayerU :: Int -> Double -> Vector Double -> [Neuron]
createSigmoidLayerU n threshold weights = 
  take n . repeat $ neuron
  where neuron = createNeuronSigmoidU threshold weights

-- | Creates a layer compound of n neurons with the Heavyside transfer
--   function, all having the given threshold and weights.
createHeavysideLayerU :: Int -> Double -> Vector Double -> [Neuron]
createHeavysideLayerU n threshold weights =
  take n . repeat $ neuron
  where neuron = createNeuronSigmoidU threshold weights

-- | Creates a layer compound of n neurons with the sigmoid transfer function,
--   all having the given threshold and weights.
createSigmoidLayer :: Int -> Double -> [Double] -> [Neuron]
createSigmoidLayer n threshold = createSigmoidLayerU n threshold . fromList

-- | Creates a layer compound of n neurons with the sigmoid transfer function,
--   all having the given threshold and weights.
createHeavysideLayer :: Int -> Double -> [Double] -> [Neuron]
createHeavysideLayer n threshold = createHeavysideLayerU n threshold . fromList

-- * Computation

-- | Computes the outputs of each Neuron of the layer
computeLayerU :: [Neuron] -> Vector Double -> Vector Double
computeLayerU ns inputs = fromList $ map (\n -> computeU n inputs) ns

-- | Computes the outputs of each Neuron of the layer
computeLayer :: [Neuron] -> [Double] -> [Double]
computeLayer ns = toList . computeLayerU ns . fromList

-- * Learning

-- | Trains each neuron with the given sample and the given learning ratio
learnSampleLayerU :: Double -> [Neuron] -> (Vector Double, Vector Double) -> [Neuron]
learnSampleLayerU alpha ns (xs, ys) = 
  Prelude.zipWith (\n y -> learnSampleU alpha n (xs, y)) ns $ toList ys

-- | Trains each neuron with the given sample and the given learning ratio
learnSampleLayer :: Double -> [Neuron] -> ([Double], [Double]) -> [Neuron]
learnSampleLayer alpha ns = learnSampleLayerU alpha ns . (fromList *** fromList)

-- | Trains each neuron with the given samples and the given learning ratio
learnSamplesLayerU :: Double -> [Neuron] -> [(Vector Double, Vector Double)] -> [Neuron]
learnSamplesLayerU alpha = Data.List.foldl' (learnSampleLayerU alpha)

-- | Trains each neuron with the given samples and the given learning ratio
learnSamplesLayer :: Double -> [Neuron] -> [([Double], [Double])] -> [Neuron]
learnSamplesLayer alpha ns = 
  learnSamplesLayerU alpha ns . map (fromList *** fromList)

-- * Quadratic Error

-- | Returns the quadratic error of a layer for a given sample
quadErrorU :: [Neuron] -> (Vector Double, Vector Double) -> Double
quadErrorU ns (xs, ys) = 
  (/2) $ Data.Vector.sum $ Data.Vector.zipWith (\o y -> (y - o)**2) os ys
  where os = computeLayerU ns xs

-- | Returns the quadratic error of a layer for a given sample
quadError :: [Neuron] -> ([Double], [Double]) -> Double
quadError ns = quadErrorU ns . (fromList *** fromList)
