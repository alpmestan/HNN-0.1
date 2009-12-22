module Main where
  
import AI.HNN.Net
import AI.HNN.Layer
import AI.HNN.Neuron
import Data.Array.Vector
import Control.Arrow
import Data.List
  
layer1, layer2 :: [Neuron]
layer1 = map (createNeuronSigmoid 0.5 . toU) [[0.5, 0.5, 0.5], [0.5,0.5,0.5], [0.5,0.5,0.5], [0.5, 0.5, 0.5]]
layer2 = [createNeuronSigmoid 0.5 $ toU [0.5, 0.4, 0.6, 0.3]]
 
net = [layer1, layer2]
  
net' = trainAux net . take 20 . cycle . map (toU *** toU) $ [([1, 1, 1],[0]), ([1, 0, 1],[1]), ([1, 1, 0],[1]), ([1, 0, 0],[0])]

testA = computeNet net $ toU [1, 1, 1]  
test = computeNet net' $ toU [1, 1, 1]
  
testB = backProp net (toU [1, 1, 1], toU [0])
  
test' = train net . map (toU *** toU) $ [([1, 1, 1],[0]), ([1, 0, 1],[1]), ([1, 1, 0],[1]), ([1, 0, 0],[0])]

main = do
     putStrLn . show $ testA
     putStrLn "---"
     putStrLn . show $ test
     putStrLn "---"
     putStrLn . show $ testB
     putStrLn "---"
     putStrLn . show $ test'