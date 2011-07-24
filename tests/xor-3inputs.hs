module Main where

import AI.HNN.Net
import AI.HNN.Layer
import AI.HNN.Neuron
import Control.Arrow
import Data.List

alpha, epsilon :: Double
alpha = 0.8     -- learning ratio
epsilon = 0.001 -- desired maximal bound for the quad error

layer1, layer2 :: [Neuron]
layer1 = map (createNeuronSigmoid 0.5)
  [[0.5, 0.5, 0.5],
   [0.5, 0.5, 0.5],
   [0.5, 0.5, 0.5],
   [0.5, 0.5, 0.5]]
layer2 = [createNeuronSigmoid 0.5 [0.5, 0.4, 0.6, 0.3]]

net = [layer1, layer2]

finalnet = train alpha epsilon net
  [([1, 1, 1], [0]),
   ([1, 0, 1], [1]),
   ([1, 1, 0], [1]),
   ([1, 0, 0], [0])]

good111 = computeNet finalnet [1, 1, 1]
good101 = computeNet finalnet [1, 0, 1]
good110 = computeNet finalnet [1, 1, 0]
good100 = computeNet finalnet [1, 0, 0]

main = do
     putStrLn $ "Final neural network : \n" ++ show finalnet
     putStrLn " ---- "
     putStrLn $ "Output for [1, 1, 1] (~ 0): " ++ show good111
     putStrLn $ "Output for [1, 0, 1] (~ 1): " ++ show good101
     putStrLn $ "Output for [1, 1, 0] (~ 1): " ++ show good110
     putStrLn $ "Output for [1, 0, 0] (~ 0): " ++ show good100
