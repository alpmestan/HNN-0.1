module Layer where

import Neuron

-- Computes the outputs of each Neuron of the layer
computeLayer :: [Neuron] -> [Double] -> [Double]
computeLayer ns inputs = map (\n -> compute n inputs) ns

-- Trains each neuron with the given sample
learnSampleLayer :: [Neuron] -> ([Double], [Double]) -> [Neuron]
learnSampleLayer ns (xs, ys) = zipWith (\n y -> learnSample n (xs, y)) ns ys

-- Trains each neuron with the given samples
learnSamplesLayer :: [Neuron] -> [([Double], [Double])] -> [Neuron]
learnSamplesLayer ns samples = foldl learnSampleLayer ns samples

-- Returns the quadratic error of a layer for a given sample
quadError :: [Neuron] -> ([Double], [Double]) -> Double
quadError ns (xs, ys) = let os = computeLayer ns xs
                        in (/2) $ sum $ zipWith (\o y -> (y - o)**2) os ys

