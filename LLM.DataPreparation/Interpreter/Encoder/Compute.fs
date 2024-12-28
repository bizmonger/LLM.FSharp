namespace LLM.DataPreparation.EmbeddingLayer

open System
open LLM.DataPreparation.Operations

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

    let contextVector : Attention.ContextVector.Compute =

        fun inputEmbeddings weights ->

            let result = inputEmbeddings |> Array.map (fun embedding -> weights |> dotProduct embedding)
            result

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

    module ProductOf =

        let numberAndVector(number:float) (vector:float[]) =

            let product = vector |> Array.map(fun v -> number * v)
            product