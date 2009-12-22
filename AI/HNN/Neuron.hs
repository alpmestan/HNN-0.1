-- | Neuron module, defining an artificial neuron type and the basical operations we can do on it
module AI.HNN.Neuron where

import Data.Array.Vector
import Data.List

-- * Type Definitions, type class instances and basic functions

-- | Our Artificial Neuron type
data Neuron = Neuron {
      threshold :: Double
    , weights   :: UArr Double
    , func      :: Double -> Double
    }

instance Show Neuron where
    show n = "Threshold : " ++ show (threshold n) ++ "\nWeights : " ++ show (weights n)

-- | Creates a Neuron with the given threshold, weights and transfer function
createNeuron :: Double -> UArr Double -> (Double -> Double) -> Neuron
createNeuron t ws f = Neuron { threshold = t, weights = ws, func = f }

-- | Equivalent to `createNeuron t ws heavyside'
createNeuronHeavyside :: Double -> UArr Double -> Neuron
createNeuronHeavyside t ws = createNeuron t ws heavyside

-- | Equivalent to `createNeuron t ws sigmoid'
createNeuronSigmoid :: Double -> UArr Double -> Neuron
createNeuronSigmoid t ws = createNeuron t ws sigmoid

-- | The Heavyside function
heavyside :: Double -> Double
heavyside x | x >= 0 = 1.0
heavyside _ = 0.0

-- | The Sigmoid function
sigmoid :: Double -> Double
sigmoid x = 1.0 / (1 + exp (-x))

-- | Computes the output of a given Neuron for given inputs
compute :: Neuron -> UArr Double -> Double
compute n inputs | lengthU inputs == lengthU (weights n) 
                     = func n $ sumU (zipWithU (*) (weights n) inputs) - threshold n
compute n inputs = error $ "Number of inputs != Number of weights\n" ++ show n ++ "\nInput : " ++ show inputs


-- * Neuron learning with Widrow-Hoff (Delta rule)

-- | Trains a neuron with the given sample, of the form (inputs, wanted_result) and the given learning ratio (alpha)
learnSample :: Double -> Neuron -> (UArr Double, Double) -> Neuron
learnSample alpha n (xs, y) = Neuron { 
                          threshold = threshold n
                        , weights = map_weights (weights n) (xs, y) 
                        , func = func n
                        }
    where map_weights ws (xs, y) = let s = compute n xs in
                                   zipWithU (\w_i x_i -> w_i + alpha*(y-s)*x_i) ws xs

-- | Trains a neuron with the given samples and the given learning ratio (alpha)
learnSamples :: Double -> Neuron -> [(UArr Double, Double)] -> Neuron
learnSamples alpha = foldl' (learnSample alpha)