-- | Neuron module, defining an artificial neuron type and the basical
--   operations we can do on it
module AI.HNN.Neuron where

import Data.Vector (Vector(), fromList)
import qualified Data.Vector as V (zipWith, sum, length)
import Data.List (foldl')

-- * Type Definitions, type class instances

-- | Our Artificial Neuron type
data Neuron = Neuron {
      threshold :: Double
    , weights   :: Vector Double
    , func      :: Double -> Double
    }

instance Show Neuron where
    show n = "Threshold : " ++ show (threshold n) ++ 
           "\nWeights   : " ++ show (weights n)

-- * Neuron creation

-- | Creates a Neuron with the given threshold, weights and transfer function
createNeuronU :: Double -> Vector Double -> (Double -> Double) -> Neuron
createNeuronU t ws f = Neuron { threshold = t, weights = ws, func = f }

-- | Equivalent to `createNeuronU t ws heavyside'
createNeuronHeavysideU :: Double -> Vector Double -> Neuron
createNeuronHeavysideU t ws = createNeuronU t ws heavyside

-- | Equivalent to `createNeuronU t ws sigmoid'
createNeuronSigmoidU :: Double -> Vector Double -> Neuron
createNeuronSigmoidU t ws = createNeuronU t ws sigmoid

-- | Same as createNeuronU, with a list instead of an Vector for the weights
--   (converted to Vector anyway)
createNeuron :: Double -> [Double] -> (Double -> Double) -> Neuron
createNeuron t ws f = createNeuronU t (fromList ws) f

-- | Same as createNeuronHeavysideU, with a list instead of an Vector for the
--   weights (converted to Vector anyway)
createNeuronHeavyside :: Double -> [Double] -> Neuron
createNeuronHeavyside t ws = createNeuronU t (fromList ws) heavyside

-- | Same as createNeuronSigmoidU, with a list instead of an Vector for the
--   weights (converted to Vector anyway)
createNeuronSigmoid :: Double -> [Double] -> Neuron
createNeuronSigmoid t ws = createNeuronU t (fromList ws) sigmoid

-- * Transfer functions

-- | The Heavyside function
heavyside :: Double -> Double
heavyside x | x >= 0 = 1.0
heavyside _ = 0.0

-- | The Sigmoid function
sigmoid :: Double -> Double
sigmoid x = 1.0 / (1 + exp (-x))

-- * Neuron output computation

-- | Computes the output of a given Neuron for given inputs
computeU :: Neuron -> Vector Double -> Double
computeU n inputs | V.length inputs == V.length (weights n) 
  = func n $ V.sum (V.zipWith (*) (weights n) inputs) - threshold n
computeU n inputs 
  = error $ "Number of inputs != Number of weights\n" ++ 
            show n ++ 
            "\nInput : " ++ show inputs

-- | Computes the output of a given Neuron for given inputs
compute :: Neuron -> [Double] -> Double
compute n = computeU n . fromList

-- * Neuron learning with Widrow-Hoff (Delta rule)

-- | Trains a neuron with the given sample, of the form (inputs, wanted_result)
--   and the given learning ratio (alpha)
learnSampleU :: Double -> Neuron -> (Vector Double, Double) -> Neuron
learnSampleU alpha n (xs, y) = Neuron { 
                          threshold = threshold n
                        , weights = map_weights (weights n) (xs, y) 
                        , func = func n
                        }
    where s = computeU n xs
          f w_i x_i = w_i + alpha*(y-s)*x_i
          map_weights ws (xs, y) = V.zipWith f ws xs

learnSample :: Double -> Neuron -> ([Double], Double) -> Neuron
learnSample alpha n (xs, y) = learnSampleU alpha n (fromList xs, y)

-- | Trains a neuron with the given samples and the given learning ratio (alpha)
learnSamplesU :: Double -> Neuron -> [(Vector Double, Double)] -> Neuron
learnSamplesU alpha = foldl' (learnSampleU alpha)

-- | Trains a neuron with the given samples and the given learning ratio (alpha)
learnSamples :: Double -> Neuron -> [([Double], Double)] -> Neuron
learnSamples alpha n samples = 
  learnSamplesU alpha n $ 
    map (\(xs, y) -> (fromList xs, y)) samples
