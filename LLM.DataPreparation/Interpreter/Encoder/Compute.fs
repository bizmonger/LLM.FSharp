namespace LLM.DataPreparation.EmbeddingLayer

open System
open System.Collections.Generic
open LLM.DataPreparation.Language
open LLM.DataPreparation.Operations
open LLM.DataPreparation

module Compute =

    let vectorSum (vectorA:float array) (vectorB:float array) : float array =

        let result = vectorA |> Array.mapi(fun i _ -> vectorA.[i] + vectorB.[i])
        result

    let vectorProduct (vectorA:float array) (vectorB:float array) : float array =

        if vectorA.Length = vectorB.Length then

            let result = vectorA |> Array.mapi(fun i v -> vectorA.[i] * vectorB.[i])
            result

        else
            let vectorLarge,vectorOther =

                if vectorA.Length >= vectorB.Length
                then vectorA,vectorB
                else vectorB,vectorA

            let mutable items = []

            for i = 0 to (vectorLarge.Length - 1) do

                for k = 0 to (vectorOther.Length - 1) do

                    let result = vectorLarge.[i] * vectorOther.[k]
                    items <- items @ [result]

            items |> List.toArray

    let multiplyAndSumVectors (vectorA:float array) (vectorB:float array) : float =

        let result = vectorA |> Array.mapi(fun i v -> vectorA.[i] * vectorB.[i]) |> Array.sum
        result

    let attentionScores : Attention.Scores.Compute =

        fun inputEmbeddings queryVector ->

            let scores = inputEmbeddings |> Array.map (fun embedding -> queryVector |> multiplyAndSumVectors embedding)
            scores

    let contextVector : Attention.ContextVector.Compute =

        fun inputEmbeddings weights ->

            let mutable items = []

            for i = 0 to (inputEmbeddings.Length - 1) do

                let item : WeightAndInputEmbedding = {
                    Weight = weights.[i]
                    InputEmbedding = inputEmbeddings.[i]
                }

                items <- [item] |> List.append items

            let result = items |> List.map(fun item -> { WeightAndInputEmbedding = item; 
                                                         Product = vectorProduct [|item.Weight|] item.InputEmbedding 
                                                       })

            let products = result |> List.map(fun v -> v.Product)
            let arraySize = products.Head.Length
            let seed : float array = Array.zeroCreate arraySize
            let sum = products |> List.fold (fun acc v -> vectorSum v acc) seed

            sum

    /// Optimized version using Span<T> for potential performance improvements.
    let attentionWeights : Attention.Weights.Compute =

        fun scores ->

            // 1. Handle empty input case.
            if scores.Length = 0 then
                [||]

            else
                // 2. Find the maximum value in the input span to prevent overflow during exponentiation.
                let maxVal = 
                    if scores.Length > 0 then
                        // 2a. Initialize max with the first element.
                        let mutable max = scores[0]
                        // 2b. Iterate through the span to find the true maximum.
                        for i = 1 to scores.Length - 1 do
                            if scores[i] > max then max <- scores[i]
                        max
                    else
                        // 2c. Handle empty input case for maxVal
                        0.0 

                // 3. Create an array to store the exponentials.
                let expValues = Array.zeroCreate<float> scores.Length 

                // 4. Calculate the exponentials of (x - maxVal) for each element x in the input span.
                for i = 0 to scores.Length - 1 do
                    expValues.[i] <- Math.Exp(scores.[i] - maxVal)

                // 5. Calculate the sum of the exponentials.
                let sumExp = expValues |> Array.sum

                // 6. Check if the sum of exponentials is zero.
                if sumExp = 0.0 then
                    // 6a. If the sum is zero, create a uniform distribution.
                    let length = scores.Length
                    Array.init length (fun _ -> 1.0 / (float length))
                else
                    // 7. Create an array to store the softmax probabilities.
                    let result = Array.zeroCreate<float> scores.Length
                    // 8. Divide each exponential by the sum to get the softmax probabilities.
                    for i = 0 to scores.Length - 1 do
                        result.[i] <- expValues.[i] / sumExp
                    result

    let contextVectorDetails (tokenToText:Dictionary<Token, TokenedText>) (inputEmbeddings:TokenEmbedding array) : QueryProduct list =

        let mutable queryProducts = []

        for queryIndex = 0 to (inputEmbeddings.Length - 1) do

            let queryVector   = inputEmbeddings.[queryIndex]
            let scores        = attentionScores inputEmbeddings queryVector
            let weights       = attentionWeights scores
            let contextVector = contextVector inputEmbeddings weights

            let queryProduct : QueryProduct = {

                Text      = tokenToText.[queryIndex]
                Token     = queryIndex
                InputEmbedding = inputEmbeddings.[queryIndex]
                Scores    = scores
                Weights   = weights
                ContextVector = contextVector
            }

            queryProducts <- queryProducts @ [queryProduct]

        queryProducts

    module ProductOf =

        let numberAndVector(number:float) (vector:float[]) =

            let product = vector |> Array.map(fun v -> number * v)
            product