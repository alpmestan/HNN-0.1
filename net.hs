module Main where

import Layer
import Neuron
import Data.List

check :: [[Neuron]] -> Bool
check nss = let l = length nss in l > 1 && l < 3

nn :: [[Neuron]] -> [[Neuron]]
nn nss | check nss = nss
       | otherwise = error "Invalided nn"

--- computeLayer is [Neuron] -> [Double] -> [Double]
--- Neuron is { threshold :: Double, weight :: [Double], func :: Double -> Double o

-- Computes the output of the given neural net
computeNet :: [[Neuron]] -> [Double] -> [Double]
computeNet neuralss xs = let nss = nn neuralss in computeLayer (nss !! 1) $ computeLayer (nss !! 0) xs
                 
quadErrorNet :: [[Neuron]] -> ([Double], [Double]) -> Double
quadErrorNet nss (xs,ys) = (sum . zipWith (\y s -> (y - s)**2) ys $ computeNet nss xs)/2.0

globalQuadErrorNet :: [[Neuron]] -> [([Double], [Double])] -> Double
globalQuadErrorNet nss samples = sum $ map (quadErrorNet nss) samples

backProp :: [[Neuron]] -> ([Double], [Double]) -> [[Neuron]]
backProp nss (xs, ys) = [aux (nss !! 0) ds_hidden xs
                        ,aux (nss !! 1) ds_out output_hidden]
    where 
      output_hidden = computeLayer (head nss) xs
      output_out = computeLayer (nss !! 1) output_hidden
      ds_out = zipWith (\s y -> s * (1 - s) * (y - s)) output_out ys
      ds_hidden = zipWith (\x s -> x * (1-x) * s) output_hidden (map (\ws -> sum $ zipWith (*) ds_out ws) (transpose . map weights $ nss !! 1))
      aux ns ds xs = zipWith (\n d -> n { weights = zipWith (\w x -> w + alpha * d * x) (weights n) xs }) ns ds

trainAux :: [[Neuron]] -> [([Double], [Double])] -> [[Neuron]]
trainAux = foldl' backProp

train :: [[Neuron]] -> [([Double], [Double])] -> [[Neuron]]
train nss samples = until (\nss' -> globalQuadErrorNet nss' samples < (0.1 :: Double)) (\nss' -> trainAux nss' samples) nss

{- TEST -}

test_net :: [[Neuron]]
test_net = [[createNeuronSigmoid 0.5 [0.1, -0.1], createNeuronSigmoid 0.5 [0.1, -0.1]], [createNeuronSigmoid 0.5 [0.5, 0.5]]]

samples :: [([Double], [Double])]
samples = [([0.0, 0.0], [0.0])
          ,([1.0, 1.0], [0.0])
          ,([1.0, 0.0], [1.0])
          ,([0.0, 1.0], [1.0])]

final_net = train test_net samples

test_output = [computeNet final_net [0.0, 0.0], computeNet final_net [1.0, 1.0]
              ,computeNet final_net [1.0, 0.0], computeNet final_net [0.0, 1.0]]

main = do
  putStrLn "Test output"
  putStrLn . show $ test_output
  putStrLn "-------------------"
  putStrLn "Final net"
  putStrLn . show $ final_net

{-

final_net == test_net

-}

