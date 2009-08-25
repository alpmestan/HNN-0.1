module Neuron where


--- Basical Definitions ---

-- Our Artificial Neuron type
data Neuron = Neuron {
      threshold :: Double
    , weights  :: [Double]
    , func     :: Double -> Double
    }

instance Show Neuron where
    show n = "Threshold : " ++ show (threshold n) ++ "\nWeights : " ++ show (weights n)

-- Creates a Neuron
createNeuron :: Double -> [Double] -> (Double -> Double) -> Neuron
createNeuron t ws f = Neuron { threshold = t, weights = ws, func = f }

-- Creates a Neuron with Heavyside as transfer function
createNeuronHeavyside :: Double -> [Double] -> Neuron
createNeuronHeavyside t ws = createNeuron t ws heavyside

-- Creates a Neuron with Sigmoid as transfer function
createNeuronSigmoid :: Double -> [Double] -> Neuron
createNeuronSigmoid t ws = createNeuron t ws sigmoid

-- The Heavyside function
heavyside :: Double -> Double
heavyside x | x >= 0 = 1.0
heavyside _ = 0.0

-- The Sigmoid function
sigmoid :: Double -> Double
sigmoid x = 1.0 / (1 + (exp $ -x))

-- Computes the output of a given Neuron for given inputs
compute :: Neuron -> [Double] -> Double
compute n inputs | length inputs == (length $ weights n) 
                     = func n $ (sum $ zipWith (*) (weights n) inputs) - (threshold n)
compute _ inputs = error $ "Number of inputs != Number of weights\n" ++ show n ++ "\nInput : " ++ show inputs


--- Single Neuron learning with Widrow-Hoff (Delta rule) ---

alpha :: Double
alpha = 0.8

-- Trains a neuron with the given sample
learnSample :: Neuron -> ([Double], Double) -> Neuron
learnSample n (xs, y) = Neuron { 
                          threshold = threshold n
                        , weights = map_weights (weights n) (xs, y) 
                        , func = func n
                        }
    where map_weights ws (xs, y) = let s = compute n xs in
                                   zipWith (\w_i x_i -> w_i + alpha*(y-s)*x_i) ws xs

-- Trains a neuron with the given samples
learnSamples :: Neuron -> [([Double], Double)] -> Neuron
learnSamples = foldl learnSample

-- Quadratic Error of the neuron w.r.t the given samples
--quadError :: Neuron -> [([Double], Double)] -> Double
--quadError n inputs = foldl acc 0.0 inputs 
--    where acc err (xs, y) = err + ((y - (compute n xs))**2.0)

-- Learning the OR function
test = [([1.0, 1.0], 1.0), ([0.0, 0.0], 0.0), ([1.0, 0.0], 1.0), ([0.0, 1.0], 0.0)]
n = createNeuronHeavyside 0.2 [0.5, 0.5]
trained = learnSamples n test

-- Learning the null function
test2 = [([1.0, 1.0], 0.0), ([0.0, 0.0], 0.0), ([1.0, 0.0], 0.0), ([0.0, 1.0], 0.0)]
n2 = createNeuronHeavyside 0.1 [0.5, 0.5]
trained2 = learnSamples n2 test2

