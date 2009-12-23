-- | Layer module, defining functions to work on a neural network layer, which is a list of neurons
module AI.HNN.Layer where

import AI.HNN.Neuron
import Control.Arrow
import Data.Array.Vector
import Data.List

-- * Layer creation

-- | Creates a layer compound of n neurons with the Sigmoid transfer function, all having the given threshold and weights.
createSigmoidLayerU :: Int -> Double -> UArr Double -> [Neuron]
createSigmoidLayerU n threshold weights = 
             let neuron = createNeuronSigmoidU threshold weights in
             take n . repeat $ neuron

-- | Creates a layer compound of n neurons with the Heavyside transfer function, all having the given threshold and weights.
createHeavysideLayerU :: Int -> Double -> UArr Double -> [Neuron]
createHeavysideLayerU n threshold weights =
             let neuron = createNeuronSigmoidU threshold weights in 
             take n . repeat $ neuron

-- | Creates a layer compound of n neurons with the sigmoid transfer function, all having the given threshold and weights.
createSigmoidLayer :: Int -> Double -> [Double] -> [Neuron]
createSigmoidLayer n threshold = createSigmoidLayerU n threshold . toU

-- | Creates a layer compound of n neurons with the sigmoid transfer function, all having the given threshold and weights.
createHeavysideLayer :: Int -> Double -> [Double] -> [Neuron]
createHeavysideLayer n threshold = createHeavysideLayerU n threshold . toU

-- * Computation

-- | Computes the outputs of each Neuron of the layer
computeLayerU :: [Neuron] -> UArr Double -> UArr Double
computeLayerU ns inputs = toU $ map (\n -> computeU n inputs) ns

-- | Computes the outputs of each Neuron of the layer
computeLayer :: [Neuron] -> [Double] -> [Double]
computeLayer ns = fromU . computeLayerU ns . toU

-- * Learning

-- | Trains each neuron with the given sample and the given learning ratio
learnSampleLayerU :: Double -> [Neuron] -> (UArr Double, UArr Double) -> [Neuron]
learnSampleLayerU alpha ns (xs, ys) = zipWith (\n y -> learnSampleU alpha n (xs, y)) ns (fromU ys)

-- | Trains each neuron with the given sample and the given learning ratio
learnSampleLayer :: Double -> [Neuron] -> ([Double], [Double]) -> [Neuron]
learnSampleLayer alpha ns = learnSampleLayerU alpha ns . (toU *** toU)

-- | Trains each neuron with the given samples and the given learning ratio
learnSamplesLayerU :: Double -> [Neuron] -> [(UArr Double, UArr Double)] -> [Neuron]
learnSamplesLayerU alpha = foldl' (learnSampleLayerU alpha)

-- | Trains each neuron with the given samples and the given learning ratio
learnSamplesLayer :: Double -> [Neuron] -> [([Double], [Double])] -> [Neuron]
learnSamplesLayer alpha ns = learnSamplesLayerU alpha ns . map (toU *** toU)

-- * Quadratic Error

-- | Returns the quadratic error of a layer for a given sample
quadErrorU :: [Neuron] -> (UArr Double, UArr Double) -> Double
quadErrorU ns (xs, ys) = let os = computeLayerU ns xs
                        in (/2) $ sumU $ zipWithU (\o y -> (y - o)**2) os ys

-- | Returns the quadratic error of a layer for a given sample
quadError :: [Neuron] -> ([Double], [Double]) -> Double
quadError ns = quadErrorU ns . (toU *** toU)
