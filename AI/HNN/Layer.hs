-- | Layer module, defining functions to work on a neural network layer, which is a list of neurons
module AI.HNN.Layer where

import AI.HNN.Neuron
import Data.Array.Vector
import Data.List


-- | Computes the outputs of each Neuron of the layer
computeLayer :: [Neuron] -> UArr Double -> UArr Double
computeLayer ns inputs = toU . map (\n -> compute n inputs) $ ns

-- | Trains each neuron with the given sample and the given learning ratio
learnSampleLayer :: Double -> [Neuron] -> (UArr Double, UArr Double) -> [Neuron]
learnSampleLayer alpha ns (xs, ys) = zipWith (\n y -> learnSample alpha n (xs, y)) ns (fromU ys)

-- | Trains each neuron with the given samples and the given learning ratio
learnSamplesLayer :: Double -> [Neuron] -> [(UArr Double, UArr Double)] -> [Neuron]
learnSamplesLayer alpha = foldl' (learnSampleLayer alpha)

-- | Returns the quadratic error of a layer for a given sample
quadError :: [Neuron] -> (UArr Double, UArr Double) -> Double
quadError ns (xs, ys) = let os = computeLayer ns xs
                        in (/2) $ sumU $ zipWithU (\o y -> (y - o)**2) os ys

