namespace LLM.DataPreparation.EmbeddingLayer

open LLM.DataPreparation.Operations
open System

module Compute =

    let vectorSum (vectorA:float array) (vectorB:float array) : float array =

        let result = vectorA |> Array.mapi(fun i v -> vectorA.[i] + vectorB.[i])
        result

    let vectorProduct (vectorA:float array) (vectorB:float array) : float array =

        let result = vectorA |> Array.mapi(fun i v -> vectorA.[i] * vectorB.[i])
        result

    let dotProduct (vectorA:float array) (vectorB:float array) : float =

        let result = vectorA |> Array.mapi(fun i v -> vectorA.[i] * vectorB.[i]) |> Array.sum
        result

    let attentionScores : Attention.Scores.Compute =

        fun inputEmbeddings queryVector ->

            let scores = inputEmbeddings |> Array.map (fun embedding -> queryVector |> dotProduct embedding)
            scores

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

    ///// Softmax function for attention weights, ensuring positivity.
    //let attentionWeights : Attention.Weights.Compute =

    //    fun scores ->

    //        // 1. Find the maximum value in the input array to prevent overflow.
    //        let maxVal = Array.max scores

    //        // 2. Calculate the exponentials, shifting by the maximum value.
    //        let expValues = scores |> Array.map (fun x -> Math.Exp(x - maxVal))

    //        // 3. Calculate the sum of the exponentials.
    //        let sumExp = Array.sum expValues

    //        // 4. Divide each exponential by the sum to get the softmax probabilities.
    //        if sumExp = 0.0 then

    //            // Handle the case where all inputs are very negative (resulting in expValues all being 0)
    //            // Returning a uniform distribution to avoid NaN or division by zero.
    //            let length = scores.Length

    //            if length = 0 
    //            then [||] // Return an empty array if the input is empty
    //            else Array.init length (fun _ -> 1.0 / (float length))

    //        else expValues |> Array.map (fun x -> x / sumExp)